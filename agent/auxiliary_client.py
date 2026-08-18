"""Shared auxiliary client router for side tasks.

Provides a single resolution chain so every consumer (context compression,
session search, web extraction, vision analysis, browser vision) picks up an
available backend without duplicating routing logic.

Resolution for text tasks (auto mode):
  1. User's main provider + main model
  2. Custom endpoint (config.yaml model.base_url + OPENAI_API_KEY)
  3. None

Resolution for vision/multimodal tasks (auto mode):
  1. Selected main provider, if it is a supported vision backend
  2. Custom endpoint
  3. None

Codex OAuth (ChatGPT-account auth) is used only when the user's main
provider *is* openai-codex, or when a caller explicitly requests it with a
model (auxiliary.<task>.provider + auxiliary.<task>.model).

Per-task overrides are configured in config.yaml under the ``auxiliary:``
section (e.g. ``auxiliary.vision.provider``, ``auxiliary.compression.model``).
Default "auto" follows the chains above.
"""

import contextlib
import contextvars
import copy
import functools
import hashlib
import inspect
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path  # noqa: F401 — used by test mocks
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse, parse_qs, urlunparse

# NOTE: `from openai import OpenAI` is deliberately NOT at module top — the
# openai SDK pulls a large type tree (~240 ms cold, including responses/*,
# graders/*). We expose `OpenAI` here as a thin proxy that imports the SDK on
# first call and forwards, so:
#   (a) the 15+ in-module `OpenAI(...)` construction sites work unchanged
#       (Python's function-scope name lookup resolves `OpenAI` to the proxy
#       object bound in module globals here, without triggering any import);
#   (b) external code can still do `auxiliary_client.OpenAI` or
#       `patch("agent.auxiliary_client.OpenAI", ...)` — tests see the proxy,
#       and patch replaces the module attribute as usual;
#   (c) `OpenAI` as a type annotation resolves at runtime to the proxy class
#       (which is harmless — annotations aren't type-checked at runtime).
# See tests/agent/test_auxiliary_client.py for patch patterns this supports.
if TYPE_CHECKING:
    from openai import OpenAI  # noqa: F401 — type hints only

_OPENAI_CLS_CACHE: Optional[type] = None


def _load_openai_cls() -> type:
    """Import and cache ``openai.OpenAI``."""
    global _OPENAI_CLS_CACHE
    if _OPENAI_CLS_CACHE is None:
        from openai import OpenAI as _cls
        _OPENAI_CLS_CACHE = _cls
    return _OPENAI_CLS_CACHE


class _OpenAIProxy:
    """Module-level proxy that looks like the ``openai.OpenAI`` class.

    Forwards ``OpenAI(...)`` calls and ``isinstance(x, OpenAI)`` checks to the
    real SDK class, importing the SDK lazily on first use.
    """

    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return _load_openai_cls()(*args, **kwargs)

    def __instancecheck__(self, obj):
        return isinstance(obj, _load_openai_cls())

    def __repr__(self):
        return "<lazy openai.OpenAI proxy>"


OpenAI = _OpenAIProxy()  # module-level name, resolves lazily on call/isinstance


# ── Availability probe mode ───────────────────────────────────────────────
# check_fns (tool gating) only need to know whether a client is RESOLVABLE —
# credentials present, provider routable. Building a real SDK client for that
# answer forces the `openai` import (~0.3s) plus httpx/SSL-context setup on
# the CLI startup path, twice (vision + browser_vision), for an object that
# is immediately discarded. Inside `aux_probe_mode()` the client constructors
# return a lightweight stub instead; resolution POLICY (which provider wins,
# credential lookup, fallback order) is unchanged and stays single-owner.
# Stubs are never cached (see _store_cached_client), so runtime callers can
# never receive one.
_aux_probe_state = threading.local()


class _AuxProbeClientStub:
    """Non-functional placeholder returned while `aux_probe_mode` is active."""

    __slots__ = ("api_key", "base_url")

    def __init__(self, api_key: str = "", base_url: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url

    def __getattr__(self, name: str) -> Any:
        # Loud failure if a probe stub ever leaks into a runtime call path
        # (it must not — stubs are cache-excluded and probe-scoped).
        raise RuntimeError(
            f"_AuxProbeClientStub used as a real client (attribute {name!r}); "
            "aux_probe_mode is for availability checks only"
        )

    def __repr__(self) -> str:
        return "<aux availability-probe client stub>"


def _aux_probe_active() -> bool:
    return bool(getattr(_aux_probe_state, "active", False))


@contextlib.contextmanager
def aux_probe_mode():
    """Resolve provider availability without constructing real SDK clients."""
    prev = getattr(_aux_probe_state, "active", False)
    _aux_probe_state.active = True
    try:
        yield
    finally:
        _aux_probe_state.active = prev

from agent.credential_pool import load_pool
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH, get_model_context_length
from pilotage_cli.config import get_pilotage_home
from utils import base_url_host_matches, base_url_hostname, env_float, is_truthy_value, model_forces_max_completion_tokens, normalize_proxy_env_vars

logger = logging.getLogger(__name__)


# ── resolve_provider_client fall-through dedup ───────────────────────────
# Both fall-through warning sites in resolve_provider_client (the "unknown
# provider" and "unhandled auth_type" branches) fire on every retry of a
# misconfigured provider, spamming the logs. Demote them to logger.debug with
# per-process dedup: the FIRST occurrence still surfaces (it carries real
# diagnostic value — a provider-name typo or PROVIDER_REGISTRY/auth_type
# drift), and identical repeats are suppressed for the lifetime of the
# process. Two independent sets keep each branch linear and let tests clear
# them independently.
_LOGGED_UNKNOWN_PROVIDER_KEYS: set = set()
_LOGGED_UNHANDLED_AUTHTYPE_KEYS: set = set()
# Same treatment for the two "registered provider, unsupported sub-branch"
# routing dead-ends — external-process and OAuth providers that fall through
# with no matching handler. Keyed by provider name.
_LOGGED_UNSUPPORTED_EXTPROC_KEYS: set = set()
_LOGGED_UNSUPPORTED_OAUTH_KEYS: set = set()


def _resolve_aux_verify(base_url: Optional[str]) -> Any:
    """Resolve httpx ``verify`` for an auxiliary-client base_url.

    Mirrors the main client's TLS resolution so auxiliary calls (compression,
    vision, web_extract, title generation, etc.) honor per-provider
    ``ssl_ca_cert`` / ``ssl_verify`` config and the ``PILOTAGE_CA_BUNDLE`` /
    ``SSL_CERT_FILE`` env conventions. Best-effort: any failure falls back to
    the httpx/certifi default (``True``).
    """
    try:
        from agent.ssl_verify import resolve_httpx_verify
        from pilotage_cli.config import (
            get_custom_provider_tls_settings,
            load_config_readonly,
        )

        tls = get_custom_provider_tls_settings(
            str(base_url or ""), config=load_config_readonly()
        )
        return resolve_httpx_verify(
            ca_bundle=tls.get("ssl_ca_cert"),
            ssl_verify=tls.get("ssl_verify"),
            base_url=str(base_url or ""),
        )
    except Exception:
        return True


_WARNED_KEEPALIVE_IMPORT_SKEW = False


def _openai_http_client_kwargs(
    base_url: Optional[str],
    *,
    async_mode: bool = False,
) -> Dict[str, Any]:
    """Inject keepalive httpx client with env-only proxy (not macOS system proxy)."""
    try:
        from agent.process_bootstrap import build_keepalive_http_client
        client = build_keepalive_http_client(
            str(base_url or ""),
            async_mode=async_mode,
            verify=_resolve_aux_verify(base_url),
        )
    except (ImportError, AttributeError):
        # Version-skewed installs: a process whose sys.path resolves
        # an older agent/process_bootstrap.py without this helper — seen when
        # the Desktop app's bundled runtime lags a git-installed source tree
        # that newer callers (cron scheduler) were written against. Every cron
        # job died on this ImportError before any agent logic ran. Degrade
        # gracefully to the OpenAI SDK's default httpx client (respects macOS
        # system proxy, no pool-level keepalive expiry) instead of failing the
        # whole job, and say so once — silent version skew is how this bug
        # went unnoticed until jobs were already dead on arrival.
        global _WARNED_KEEPALIVE_IMPORT_SKEW
        if not _WARNED_KEEPALIVE_IMPORT_SKEW:
            _WARNED_KEEPALIVE_IMPORT_SKEW = True
            logger.warning(
                "agent.process_bootstrap.build_keepalive_http_client is "
                "unavailable — mixed/stale install detected. Falling "
                "back to the SDK default HTTP client. Run `pilotage update` (or "
                "reinstall the Desktop app) to resync the runtime."
            )
        client = None

    if client is None:
        return {}
    return {"http_client": client}

def _create_openai_client(*, api_key: str, base_url: str, **kwargs: Any) -> Any:
    if _aux_probe_active():
        # Availability probe: credentials/base_url resolved — that is the
        # answer. Skip the openai import + httpx/SSL construction entirely.
        return _AuxProbeClientStub(api_key=api_key, base_url=base_url)
    kwargs = {**_openai_http_client_kwargs(base_url), **kwargs}
    # Pilotage owns auxiliary retry + provider/model fallback policy (the
    # same-provider transient retry in call_llm plus the except-chain
    # fallback). The OpenAI SDK's own default (max_retries=2 → up to 3
    # attempts) silently multiplies the effective wall time of every aux call
    # by 3× on a slow/hung endpoint, so a 120s timeout can stall ~360s before
    # Pilotage sees a single failure. Disable SDK-internal retries
    # by default and let Pilotage control the budget; explicit callers can still
    # override via kwargs.
    kwargs.setdefault("max_retries", 0)
    return OpenAI(api_key=api_key, base_url=base_url, **kwargs)


# ── Interrupt protection for atomic auxiliary tasks ──────────────────────
# Some auxiliary tasks must NOT be aborted mid-flight by a gateway interrupt
# (e.g. an incoming user message while the agent is busy). Context
# compression is the prime case: if the summary LLM call is interrupted
# part-way, compression falls back to a static "summary unavailable" marker
# and the real handoff is lost. A thread-local flag lets such a
# task mark its in-flight LLM call as interrupt-protected; the Codex
# Responses stream's cancellation check honors it. An explicit host cancel
# (CLI Ctrl+C or /stop) may install a cancel check that overrides protection;
# ordinary incoming-message interrupts remain protected. TIMEOUTS still fire
# (a hung call must die), and all OTHER aux tasks (vision, web_extract,
# title_generation, …) remain freely interruptible.
_aux_interrupt_protection = threading.local()


class AuxiliaryExplicitCancellation(BaseException):
    """Frozen signal that an auxiliary attempt was explicitly hard-cancelled.

    This deliberately follows ``asyncio.CancelledError`` and inherits directly
    from ``BaseException``: provider retry/fallback code catches ``Exception``
    broadly and must never reinterpret an explicit host stop as a transport
    failure. ``cause`` is immutable class data so downstream compression code
    does not re-query a mutable host Event after the transport has unwound.
    """

    cause = "explicit_host_cancel"

    def __init__(self) -> None:
        super().__init__("auxiliary request explicitly cancelled by host")


def _aux_interrupt_protected() -> bool:
    return bool(getattr(_aux_interrupt_protection, "active", False))


def _aux_interrupt_cancel_requested() -> bool:
    """Return whether an explicit host cancel overrides aux protection."""
    event = getattr(_aux_interrupt_protection, "cancel_event", None)
    if event is not None:
        try:
            return bool(event.is_set())
        except Exception:
            logger.debug("aux interrupt cancel event check failed", exc_info=True)
            return False
    check = getattr(_aux_interrupt_protection, "cancel_check", None)
    if not callable(check):
        return False
    try:
        return bool(check())
    except Exception:
        logger.debug("aux interrupt cancel check failed", exc_info=True)
        return False


@contextlib.contextmanager
def aux_interrupt_protection(
    active: bool = True,
    cancel_check=None,
    cancel_event=None,
):
    """Mark the current thread's auxiliary LLM call as interrupt-protected.

    Used by atomic aux tasks (compression) so a mid-flight gateway interrupt
    doesn't abort the call and trigger a degraded fallback. Re-entrant-safe:
    restores the previous value on exit. ``cancel_check`` lets the host retain
    an explicit hard-cancel path; ``cancel_event`` is preferred when the host
    already owns an Event. Nested protection scopes inherit both values.
    """
    prev = getattr(_aux_interrupt_protection, "active", False)
    prev_cancel_check = getattr(_aux_interrupt_protection, "cancel_check", None)
    prev_cancel_event = getattr(_aux_interrupt_protection, "cancel_event", None)
    _aux_interrupt_protection.active = active
    if callable(cancel_check):
        _aux_interrupt_protection.cancel_check = cancel_check
    if cancel_event is not None and callable(getattr(cancel_event, "is_set", None)):
        _aux_interrupt_protection.cancel_event = cancel_event
    try:
        yield
    finally:
        _aux_interrupt_protection.active = prev
        _aux_interrupt_protection.cancel_check = prev_cancel_check
        _aux_interrupt_protection.cancel_event = prev_cancel_event


def _capture_aux_cancel_check() -> Optional[Callable[[], Any]]:
    """Capture the current explicit-cancel source on the owning request thread."""
    event = getattr(_aux_interrupt_protection, "cancel_event", None)
    is_set = getattr(event, "is_set", None)
    if callable(is_set):
        return is_set
    check = getattr(_aux_interrupt_protection, "cancel_check", None)
    if callable(check):
        # Preserve callable identity so attempt-local decision objects retain
        # methods such as begin_timeout_cleanup() when captured by adapters.
        return check
    return None


def _captured_aux_cancel_requested(cancel_check: Callable[[], Any]) -> bool:
    """Read a request-thread cancellation source without leaking its failures."""
    try:
        return bool(cancel_check())
    except Exception:
        logger.debug("captured aux cancel check failed", exc_info=True)
        return False


class _AuxiliaryCancellationDecision:
    """Atomically choose explicit cancellation or provider timeout per attempt."""

    def __init__(self, source_cancel_check: Callable[[], Any]) -> None:
        self._source_cancel_check = source_cancel_check
        self._lock = threading.Lock()
        self._outcome = "active"

    def __call__(self) -> bool:
        with self._lock:
            if self._outcome == "cancelled":
                return True
            if self._outcome == "timed_out":
                return False
            if _captured_aux_cancel_requested(self._source_cancel_check):
                self._outcome = "cancelled"
                return True
            return False

    def begin_timeout_cleanup(self) -> bool:
        """Return whether timeout won and destructive cleanup is permitted."""
        with self._lock:
            if self._outcome == "active":
                if _captured_aux_cancel_requested(self._source_cancel_check):
                    self._outcome = "cancelled"
                else:
                    self._outcome = "timed_out"
            return self._outcome == "timed_out"


# ── Forward-progress hook for streamed auxiliary calls ───────────────────
# Long auxiliary calls (context compression is the prime case) are watched by
# wall-clock deadlines in their hosts (gateway session hygiene). A fixed
# deadline punishes SLOW summary models exactly as hard as HUNG ones: a
# reasoning model happily streaming a large summary is killed mid-generation.
# This thread-local hook lets the host observe liveness instead: the wire
# consumers below tick it on every streamed token/SSE event, and the host
# extends its deadline while tokens are moving (see gateway/run.py session
# hygiene + CompressionCommitFence.touch_progress). Thread-local matches the
# call topology — the aux call and its stream consumption run synchronously
# on the thread that installed the hook.
_aux_progress = threading.local()


def _notify_aux_progress() -> None:
    """Tick the installed forward-progress hook, if any. Never raises."""
    hook = getattr(_aux_progress, "hook", None)
    if hook is None:
        return
    try:
        hook()
    except Exception:
        logger.debug("aux progress hook failed", exc_info=True)


def _aux_progress_active() -> bool:
    return getattr(_aux_progress, "hook", None) is not None


@contextlib.contextmanager
def aux_progress_hook(hook):
    """Install *hook* as the current thread's aux forward-progress callback.

    ``hook=None`` is a no-op passthrough so callers can wire it
    unconditionally. Re-entrant-safe: restores the previous hook on exit.
    """
    prev = getattr(_aux_progress, "hook", None)
    _aux_progress.hook = hook if callable(hook) else prev
    try:
        yield
    finally:
        _aux_progress.hook = prev


def _run_protected_sync_provider_call(
    callback: Callable[[dict[str, Any]], Any],
    kwargs: dict[str, Any],
) -> Any:
    """Run one protected provider callback in an attempt-isolated daemon.

    A hard cancel must release the compression-owning thread promptly, but
    auxiliary clients are process-shared and cannot safely be closed or evicted
    to wake one request.  Only protected calls with a captured hard-cancel source
    use this seam.  Their provider callback (including stream aggregation) runs
    in a daemon worker while the owner polls cancellation.  On cancel the owner
    unwinds immediately; the worker is left to finish under the provider timeout
    already present in ``kwargs``.  It owns no transcript or compressor commit
    state and never holds the session lock.

    Ordinary auxiliary calls, and protected calls without a cancellation source,
    retain the historical direct synchronous path with no extra thread.
    """
    source_cancel_check = _capture_aux_cancel_check()
    if not _aux_interrupt_protected() or not callable(source_cancel_check):
        return callback(kwargs)

    # Freeze one linearized outcome for this isolated attempt. The host Event is
    # reused and cleared on a later turn, while the Codex timeout Timer may race
    # owner polling. Both paths must decide under the same attempt-local lock.
    cancel_check = _AuxiliaryCancellationDecision(source_cancel_check)

    if cancel_check():
        raise AuxiliaryExplicitCancellation()

    progress_hook = getattr(_aux_progress, "hook", None)
    provider_context = contextvars.copy_context()
    done = threading.Event()
    outcome: dict[str, Any] = {}

    def _provider_worker() -> None:
        try:
            with aux_progress_hook(progress_hook), aux_interrupt_protection(
                cancel_check=cancel_check
            ):
                outcome["result"] = callback(kwargs)
        except BaseException as exc:
            outcome["exception"] = exc
        finally:
            done.set()

    threading.Thread(
        target=provider_context.run,
        args=(_provider_worker,),
        name="pilotage-protected-aux-provider",
        daemon=True,
    ).start()

    while True:
        # Cancellation is checked before and after every completion wait so it
        # wins whenever result publication and the host Event become visible in
        # the same polling interval.
        if _captured_aux_cancel_requested(cancel_check):
            raise AuxiliaryExplicitCancellation()
        if not done.wait(0.02):
            continue
        if _captured_aux_cancel_requested(cancel_check):
            raise AuxiliaryExplicitCancellation()
        exception = outcome.get("exception")
        if exception is not None:
            raise exception
        return outcome.get("result")


def _safe_isinstance(obj: Any, maybe_type: Any) -> bool:
    """Return False instead of raising when a patched symbol is not a type."""
    try:
        return isinstance(obj, maybe_type)
    except TypeError:
        return False


def _extract_url_query_params(url: str):
    """Extract query params from URL, return (clean_url, default_query dict or None)."""
    parsed = urlparse(url)
    if parsed.query:
        clean = urlunparse(parsed._replace(query=""))
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        return clean, params
    return url, None


# Module-level flag: only warn once per process about stale OPENAI_BASE_URL.
_stale_base_url_warned = False

_PROVIDER_ALIASES: dict = {}


def _normalize_aux_provider(provider: Optional[str]) -> str:
    normalized = (provider or "auto").strip().lower()
    if normalized.startswith("custom:"):
        suffix = normalized.split(":", 1)[1].strip()
        if not suffix:
            return "custom"
        normalized = suffix
    if normalized == "codex":
        return "openai-codex"
    if normalized == "main":
        # Resolve to the user's actual main provider so named custom providers
        # work correctly.
        main_prov = (_read_main_provider() or "").strip().lower()
        if main_prov and main_prov not in {"auto", "main", ""}:
            normalized = main_prov
        else:
            return "custom"
    return _PROVIDER_ALIASES.get(normalized, normalized)


# Context window enforced by ChatGPT's Codex OAuth backend for the
# gpt-5.4 / gpt-5.5 / gpt-5.6 families. The raw OpenAI API exposes 1.05M
# for the same slugs, but the Codex backend hard-caps at 272K
# (verified live for 5.4/5.5: a ~330K-token request to
# chatgpt.com/backend-api/codex/responses is rejected with
# ``context_length_exceeded`` while ~250K succeeds; gpt-5.6 shares the same
# 272K Codex cap — see _CODEX_OAUTH_CONTEXT_FALLBACK in model_metadata.py).
# With a 272K ceiling the default 50% compaction trigger fires at ~136K —
# wasteful, since the model can hold far more raw context before
# summarization actually buys anything. We raise the trigger to 85% (~231K)
# on this exact route so Codex gpt-5.4 / gpt-5.5 / gpt-5.6 sessions use the
# window they actually have.
_CODEX_GPT54_GPT55_COMPACTION_THRESHOLD = 0.85

# gpt-5.3-codex-spark is Codex-OAuth-only (ChatGPT Pro entitlement) with a
# native 128K context window.  The default 50% compaction trigger fires at
# ~64K — wasting half the usable window, often before the session has enough
# turns to summarize meaningfully.  We raise the trigger to 70% (~90K) so
# spark sessions use more of the window before summarization, while still
# leaving ~38K headroom for the summary and continued conversation before
# the 128K hard limit.
_CODEX_SPARK_COMPACTION_THRESHOLD = 0.70


def _is_codex_gpt54_or_gpt55(model: Optional[str], provider: Optional[str] = None) -> bool:
    """True for gpt-5.4 / gpt-5.5 / gpt-5.6 on the ChatGPT Codex OAuth backend.

    Matches only the Codex OAuth route (provider ``openai-codex``), not the
    direct OpenAI API path — which exposes a
    larger context window for the same slug and must keep the user's default
    compaction threshold. ``-pro`` variants and dated snapshots are matched
    via prefix so the override tracks every 272K-capped family (5.4, 5.5,
    5.6 sol/terra/luna incl. their ``-pro`` modes) without re-listing every
    variant. (Name kept for backward compatibility with the
    ``compression.codex_gpt55_autoraise`` config key.)
    """
    prov = (provider or "").strip().lower()
    if prov != "openai-codex":
        return False
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    return (
        bare == "gpt-5.4"
        or bare.startswith("gpt-5.4-")
        or bare.startswith("gpt-5.4.")
        or bare == "gpt-5.5"
        or bare.startswith("gpt-5.5-")
        or bare.startswith("gpt-5.5.")
        or bare == "gpt-5.6"
        or bare.startswith("gpt-5.6-")
        or bare.startswith("gpt-5.6.")
    )


def _is_codex_spark(model: Optional[str], provider: Optional[str] = None) -> bool:
    """True for ``gpt-5.3-codex-spark`` on the ChatGPT Codex OAuth backend.

    The model is Codex-OAuth-only (ChatGPT Pro entitlement) with a native
    128K context window.  Only the Codex OAuth route (provider
    ``openai-codex``) is matched — the slug is not available on other
    routes.
    """
    prov = (provider or "").strip().lower()
    if prov != "openai-codex":
        return False
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    return bare == "gpt-5.3-codex-spark"


def _compression_threshold_for_model(
    model: Optional[str],
    provider: Optional[str] = None,
    *,
    allow_codex_gpt55_autoraise: bool = True,
) -> Optional[float]:
    """Return a context-compression threshold override for specific models.

    The threshold is the fraction of the model's context window that must be
    consumed before Pilotage triggers summarization.  Higher values delay
    compression and preserve more raw context.

    Per-model/route overrides:
      - gpt-5.4 / gpt-5.5 / gpt-5.6 on the Codex OAuth route → 0.85, because
        Codex caps all three families at 272K and the default 50% trigger
        would compact at ~136K. Gated by ``allow_codex_gpt55_autoraise``
        (historical config-key name kept for backward compatibility) so the
        user can opt back down to the global default (the caller passes the
        config flag through here).
      - gpt-5.3-codex-spark on the Codex OAuth route → 0.70, because the model
        has a native 128K window and the default 50% trigger would compact at
        ~64K — wasting half the usable context. Not gated by the gpt-5.5
        opt-out flag: 128K is the model's native window, so the raise is
        unambiguously correct.

    Returns a float in (0, 1] to override the global ``compression.threshold``
    config value, or ``None`` to leave the user's config value unchanged.
    """
    if allow_codex_gpt55_autoraise and _is_codex_gpt54_or_gpt55(model, provider):
        return _CODEX_GPT54_GPT55_COMPACTION_THRESHOLD
    if _is_codex_spark(model, provider):
        return _CODEX_SPARK_COMPACTION_THRESHOLD
    return None

# Model-family priority for the auxiliary "fast tier", fastest first.
#
# Matched as substrings against the provider's LIVE /v1/models catalog rather
# than pinned as exact ids, because exact ids rot: a hardcoded slug starts
# 404ing the moment the provider retires it, and every aux call pays a
# wasted round-trip before the retry net catches it. Families outlive their
# version numbers, so a new mini/nano release is picked up with no edit.
#
# Rolling "-latest" aliases come first where a provider publishes them: they
# are the only ids that are structurally rot-proof.
_FAST_MODEL_FAMILIES: tuple = (
    "gpt-mini-latest",
    "gpt-nano-latest",
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-5-mini",
    "-nano",
    "-mini",
)

# Substrings that disqualify an otherwise-matching id. Reasoning variants
# ("o3-mini", "gpt-5.4-mini-thinking") think before answering, which is the
# opposite of what a titler wants; ":batch" is an async queue, not a live
# endpoint; embedding models ("all-minilm") match "-mini" but aren't chat
# models at all; ":free" tiers are heavily rate-limited and measured slowest.
# The modality suffixes are the same trap as the embedders — a provider names
# its speech and image endpoints after the chat model they're paired with, so
# "gpt-4o-mini-tts" satisfies the "-mini" rung and cannot answer a prompt.
_FAST_MODEL_EXCLUDE: tuple = (
    "thinking", "reason", "-r1", "minilm", ":batch", ":free",
    "o1-", "o3-", "o4-", "codex", "audio", "-vl", "embed",
    "-tts", "-transcribe", "-realtime", "-image", "-search-preview",
)


_VERSION_CHUNK_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _model_recency_key(model_id: str) -> tuple:
    """Sort key that puts a family's newest release first (descending).

    The rungs at the bottom of ``_FAST_MODEL_FAMILIES`` are bare family names —
    ``-mini``, ``-nano`` — and a provider serves every generation of those
    it hasn't retired. Compared as plain strings, the oldest wins:
    ``gpt-3.5-mini`` sorts before ``gpt-5.4-mini``. So the rung meant to
    keep us current on a provider's
    small tier was pinning us to its most obsolete member.

    Splitting digit runs out and comparing them as numbers fixes both the
    generation order and the 9-vs-10 cliff a string sort walks off.
    """
    chunks = []
    for index, part in enumerate(_VERSION_CHUNK_RE.split(model_id.lower())):
        if not part:
            continue
        # re.split with one capturing group alternates text, number, text, …
        chunks.append((1, float(part), "") if index % 2 else (0, 0.0, part))
    return tuple(chunks)


def _fast_model_from_catalog(provider_id: str) -> str:
    """Pick the fastest small model the provider ACTUALLY serves right now.

    Reads the provider's live (cached) ``/v1/models`` catalog and returns the
    newest ``_FAST_MODEL_FAMILIES`` match. Returns "" when the catalog is
    unavailable or holds no small model, so the caller falls through to the
    provider's curated default. Never raises and never blocks on a cold
    network path — the underlying fetch is memory+disk cached with a
    last-known-good fallback.
    """
    try:
        from pilotage_cli.auth import resolve_api_key_provider_credentials
        from pilotage_cli.models import fetch_models_with_pricing
        from providers import get_provider_profile

        # The provider's own credentials, because most ``/v1/models`` endpoints
        # are authenticated: fetched anonymously they 401, and the caller reads
        # that as "this provider serves no small model" and quietly falls back
        # to the curated default forever.
        api_key, base_url = "", ""
        try:
            creds = resolve_api_key_provider_credentials(provider_id) or {}
            api_key = str(creds.get("api_key", "")).strip()
            base_url = str(creds.get("base_url", "")).strip()
        except Exception:
            # Not an API-key provider, or nothing configured yet. The anonymous
            # fetch below still works for the catalogs that allow it.
            logger.debug("No credentials for %s catalog", provider_id, exc_info=True)

        if not base_url:
            base_url = str(getattr(get_provider_profile(provider_id), "base_url", "") or "")
        base_url = base_url.rstrip("/")
        if not base_url:
            return ""
        # fetch_models_with_pricing appends its own /v1/models.
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        catalog = fetch_models_with_pricing(
            api_key=api_key or None, base_url=base_url, timeout=3.0
        ) or {}
    except Exception:
        logger.debug("Fast-model catalog lookup failed for %s", provider_id, exc_info=True)
        return ""

    ids = sorted((str(m) for m in catalog), key=_model_recency_key, reverse=True)
    for family in _FAST_MODEL_FAMILIES:
        for model_id in ids:
            lowered = model_id.lower()
            if family in lowered and not any(x in lowered for x in _FAST_MODEL_EXCLUDE):
                return model_id
    return ""


# Default auxiliary models for direct API-key providers (cheap/fast for side tasks)
def _get_aux_model_for_provider(provider_id: str, *, prefer_fast: bool = False) -> str:
    """Return the cheap auxiliary model for a provider.

    Resolution ladder, fastest-and-most-live first:

    1. ``prefer_fast`` only — a family match against the provider's LIVE
       ``/v1/models`` catalog, preferring rolling ``-latest`` aliases. This is
       both rot-proof and latency-ordered.
    2. ``prefer_fast`` only — the provider's own recommendation hook
       (``ProviderProfile.resolve_aux_model``). Live, but tuned for *quality*
       on long-context side tasks, so it ranks below the catalog match for
       latency-critical work.
    3. ``ProviderProfile.default_aux_model`` — curated, hardcoded, may rot.
    4. The legacy hardcoded dict, for providers predating the profiles system.

    ``prefer_fast`` is opt-in so this only changes latency-critical tasks
    (titling). Every other auxiliary caller keeps the existing static
    behaviour and its cache keys.
    """
    profile = None
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(provider_id)
    except Exception:
        pass

    if prefer_fast:
        catalog_pick = _fast_model_from_catalog(provider_id)
        if catalog_pick:
            return catalog_pick
        if profile is not None:
            try:
                live = profile.resolve_aux_model()
                if live:
                    return live
            except Exception:
                logger.debug("resolve_aux_model failed for %s", provider_id, exc_info=True)

    if profile is not None and profile.default_aux_model:
        return profile.default_aux_model
    return _API_KEY_PROVIDER_AUX_MODELS_FALLBACK.get(provider_id, "")



# Fallback for providers that do not set ProviderProfile.default_aux_model.
_API_KEY_PROVIDER_AUX_MODELS_FALLBACK: Dict[str, str] = {}

# Legacy alias — callers that haven't been updated to _get_aux_model_for_provider()
# can still use this dict directly. Kept in sync with _FALLBACK above.
_API_KEY_PROVIDER_AUX_MODELS: Dict[str, str] = _API_KEY_PROVIDER_AUX_MODELS_FALLBACK

# Auxiliary tasks that may opt into the provider's fast/cheap model instead of
# the user's main chat model. The opt-in lives in
# ``auxiliary.<task>.prefer_fast_model`` so the default ``auto = main model``
# contract remains true on every settings surface.
_FAST_MODEL_TASKS: frozenset = frozenset({"title_generation"})


def _task_prefers_fast_model(task: Optional[str]) -> bool:
    """Return whether an eligible task explicitly opts into fast-model routing."""
    if task not in _FAST_MODEL_TASKS:
        return False
    task_config = _get_auxiliary_task_config(task)
    return is_truthy_value(task_config.get("prefer_fast_model"), default=False)


# Vision-specific model overrides for direct providers.
# When the user's main provider has a dedicated vision/multimodal model that
# differs from their main chat model, map it here.  The vision auto-detect
# "exotic provider" branch checks this before falling back to the main model.
_PROVIDER_VISION_MODELS: Dict[str, str] = {}


def _resolve_provider_vision_default(provider: str) -> Optional[str]:
    """Return the provider's preferred default vision model id, or None.

    Static entries in :data:`_PROVIDER_VISION_MODELS` win first. Otherwise the
    provider's :class:`ProviderProfile` supplies one via its
    ``default_vision_model()`` hook, keeping discovery inside the plugin
    instead of a name-check branch here.
    """
    static = _PROVIDER_VISION_MODELS.get(provider)
    if static:
        return static
    try:
        from providers import get_provider_profile
        profile = get_provider_profile(provider)
    except Exception:
        return None
    if profile is None:
        return None
    try:
        return profile.default_vision_model()
    except Exception:
        return None

# Providers whose endpoint does not accept image input, even though the
# provider's broader ecosystem has vision models available elsewhere.  When
# `auxiliary.vision.provider: auto` sees one of these as the main provider,
# it must skip straight to the aggregator chain instead of returning a client
# that will 404 on every vision request.
_PROVIDERS_WITHOUT_VISION: frozenset = frozenset()

def _apply_user_default_headers(headers: dict | None) -> dict | None:
    """Merge user-configured ``model.default_headers`` onto resolved headers.

    User values take precedence over provider/SDK defaults, mirroring the main
    agent client (``AIAgent._apply_user_default_headers``). This lets a
    ``custom`` OpenAI-compatible endpoint behind a gateway/WAF that rejects the
    OpenAI SDK's identifying headers (``User-Agent: OpenAI/Python ...``,
    ``X-Stainless-*``) override them for auxiliary calls too — otherwise the
    main turn would succeed but title/compression/vision calls to the same
    endpoint would still fail.

    Returns the merged dict, or the original ``headers`` (possibly ``None``)
    when nothing is configured. No allocation when there are no overrides.
    """
    try:
        from pilotage_cli.config import cfg_get, load_config
        _cfg = load_config()
        user_headers = cfg_get(_cfg, "model", "default_headers")
        # ``model.extra_headers`` is an accepted alias (matches the
        # per-provider ``extra_headers`` key on providers/custom_providers
        # entries). When both are set they merge, with ``extra_headers``
        # winning. SECURITY: values may carry credentials — never log them.
        alias_headers = cfg_get(_cfg, "model", "extra_headers")
        if isinstance(alias_headers, dict) and alias_headers:
            merged_user: dict = {}
            if isinstance(user_headers, dict):
                merged_user.update(user_headers)
            merged_user.update(alias_headers)
            user_headers = merged_user
    except Exception:
        return headers
    if not isinstance(user_headers, dict) or not user_headers:
        return headers
    merged = dict(headers or {})
    for key, value in user_headers.items():
        if value is None:
            continue
        merged[str(key)] = str(value)
    return merged or headers


_AUTH_JSON_PATH = get_pilotage_home() / "auth.json"

# Codex OAuth endpoint used when a caller explicitly requests
# provider="openai-codex".  There is deliberately no hardcoded default
# model: the set of models OpenAI accepts on this endpoint for
# ChatGPT-account auth is an undocumented, shifting allow-list, and
# pinning one here has drifted silently twice (gpt-5.3-codex → gpt-5.2-codex
# → gpt-5.4 over 6 weeks in early 2026).  Callers must pass the model
# they want explicitly (from config.yaml model.model, auxiliary.<task>.model,
# or the user's active Codex model selection).
_CODEX_AUX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _codex_cloudflare_headers(access_token: str) -> Dict[str, str]:
    """Headers required to avoid Cloudflare 403s on chatgpt.com/backend-api/codex.

    The Cloudflare layer in front of the Codex endpoint whitelists a small set of
    first-party originators (``codex_cli_rs``, ``codex_vscode``, ``codex_sdk_ts``,
    anything starting with ``Codex``). Requests from non-residential IPs (VPS,
    server-hosted agents) that don't advertise an allowed originator are served
    a 403 with ``cf-mitigated: challenge`` regardless of auth correctness.

    We pin ``originator: codex_cli_rs`` to match the upstream codex-rs CLI, set
    ``User-Agent`` to a codex_cli_rs-shaped string (beats SDK fingerprinting),
    and extract ``ChatGPT-Account-ID`` (canonical casing, from codex-rs
    ``auth.rs``) out of the OAuth JWT's ``chatgpt_account_id`` claim.

    Malformed tokens are tolerated — we drop the account-ID header rather than
    raise, so a bad token still surfaces as an auth error (401) instead of a
    crash at client construction.
    """
    headers = {
        "User-Agent": "codex_cli_rs/0.0.0 (Pilotage Agent)",
        "originator": "codex_cli_rs",
    }
    if not isinstance(access_token, str) or not access_token.strip():
        return headers
    try:
        import base64
        parts = access_token.split(".")
        if len(parts) < 2:
            return headers
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        acct_id = claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
        if isinstance(acct_id, str) and acct_id:
            headers["ChatGPT-Account-ID"] = acct_id
    except Exception:
        pass
    return headers


def _to_openai_base_url(base_url: str) -> str:
    """Normalize a configured base URL (trim whitespace and trailing slash)."""
    return str(base_url or "").strip().rstrip("/")


def _select_pool_entry(provider: str) -> Tuple[bool, Optional[Any]]:
    """Return (pool_exists_for_provider, selected_entry)."""
    try:
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug("Auxiliary client: could not load pool for %s: %s", provider, exc)
        return False, None
    if not pool or not pool.has_credentials():
        return False, None
    try:
        return True, pool.select()
    except Exception as exc:
        logger.debug("Auxiliary client: could not select pool entry for %s: %s", provider, exc)
        return True, None


def _peek_pool_entry(provider: str) -> Optional[Any]:
    """Best-effort current/next pool entry without mutating selection order."""
    try:
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug("Auxiliary client: could not load pool for %s (peek): %s", provider, exc)
        return None
    if not pool or not pool.has_credentials():
        return None
    try:
        current_fn = getattr(pool, "current", None)
        if callable(current_fn):
            current = current_fn()
            if current is not None:
                return current
        peek_fn = getattr(pool, "peek", None)
        if callable(peek_fn):
            return peek_fn()
    except Exception as exc:
        logger.debug("Auxiliary client: could not peek pool entry for %s: %s", provider, exc)
    return None


def _pool_runtime_api_key(entry: Any) -> str:
    if entry is None:
        return ""
    # Use the PooledCredential.runtime_api_key property which handles
    key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    return str(key or "").strip()


def _pool_runtime_base_url(entry: Any, fallback: str = "") -> str:
    if entry is None:
        return str(fallback or "").strip().rstrip("/")
    # Fall back through inference_base_url and base_url for non-PooledCredential entries.
    url = (
        getattr(entry, "runtime_base_url", None)
        or getattr(entry, "inference_base_url", None)
        or getattr(entry, "base_url", None)
        or fallback
    )
    return str(url or "").strip().rstrip("/")


def _scoped_key_env(name: str) -> str:
    """Read a provider API key env var through the profile secret scope.

    Auxiliary-client resolution runs both inside agent turns (secret scope
    installed — its verdict is authoritative under multiplex, so a scoped
    miss must NOT borrow another profile's process-env key) and on unscoped
    startup/CLI probe paths, which keep the legacy ``os.environ`` read via
    the ``UnscopedSecretError`` fallback (Slack pattern,).
    """
    if not name:
        return ""
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            return (get_secret(name) or "").strip()
        except UnscopedSecretError:
            pass
    except Exception:
        pass
    return (os.getenv(name) or "").strip()


# ── Codex Responses → chat.completions adapter ─────────────────────────────
# All auxiliary consumers call client.chat.completions.create(**kwargs) and
# read response.choices[0].message.content. This adapter translates those
# calls to the Codex Responses API so callers don't need any changes.


class _CodexCompletionsAdapter:
    """Drop-in shim that accepts chat.completions.create() kwargs and
    routes them through the Codex Responses streaming API."""

    def __init__(self, real_client: OpenAI, model: str):
        self._client = real_client
        self._model = model

    def create(self, **kwargs) -> Any:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)

        # Separate system/instructions from replayable conversation messages,
        # then route the rest through the SINGLE shared chat->Responses
        # converter used by the main agent transport
        # (agent/transports/codex.py). Maintaining a private conversion loop
        # here let chat-style messages with role="tool" leak straight into
        # Responses input[] — which the Responses API rejects with
        # "Invalid value: 'tool'. Supported values are: 'assistant', 'system',
        # 'developer', and 'user'." (, hit hard by flush_memories
        # / compression replaying real session history that includes assistant
        # tool_calls + role="tool" results). The shared converter encodes
        # assistant tool calls as `function_call` items and tool results as
        # `function_call_output` items with a valid call_id, so every
        # Responses path normalizes tool history identically and cannot drift.
        from agent.codex_responses_adapter import _chat_messages_to_responses_input
        from utils import base_url_host_matches

        instructions = "You are a helpful assistant."
        replay_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                instructions = content if isinstance(content, str) else str(content)
            else:
                replay_messages.append(msg)

        # Auxiliary calls never send ``context_management`` (native
        # compaction is a main-turn feature), so they must never replay a
        # compaction checkpoint from the replayed history nor let one
        # restructure this request — the summarizer/aggregator model is
        # usually not even the one that minted the blob.
        input_items = _chat_messages_to_responses_input(
            replay_messages,
            native_compaction_eligible=False,
        )

        resp_kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_items or [{"role": "user", "content": ""}],
            "store": False,
        }

        # Preserve the chat.completions timeout contract. This adapter is used
        # by auxiliary calls such as context compression; if the timeout is not
        # forwarded and enforced, a Codex Responses stream can sit behind a
        # dead-looking CLI until the user force-interrupts the whole session.
        timeout = kwargs.get("timeout")
        if timeout is not None:
            resp_kwargs["timeout"] = timeout

        # Note: the Codex endpoint (chatgpt.com/backend-api/codex) does NOT
        # support max_output_tokens or temperature — omit to avoid 400 errors.

        # Translate extra_body.reasoning (chat.completions shape) into the
        # Responses API's top-level reasoning + include fields.  Mirrors
        # agent/transports/codex.py::build_kwargs() so auxiliary callers
        # that configure reasoning via auxiliary.<task>.extra_body get the
        # same behavior as the main agent's Codex transport.
        extra_body = kwargs.get("extra_body") or {}
        if isinstance(extra_body, dict):
            reasoning_cfg = extra_body.get("reasoning")
            if isinstance(reasoning_cfg, dict):
                if reasoning_cfg.get("enabled") is False:
                    # Reasoning explicitly disabled — do not set reasoning
                    # or include.  The Codex backend still thinks by
                    # default, but we honor the caller's intent where the
                    # API allows it.
                    pass
                else:
                    # Truthy-only check mirrors agent/transports/codex.py
                    # build_kwargs(): falsy values (None, "", 0) fall back
                    # to the default rather than being forwarded to the
                    # Codex backend, which rejects e.g. {"effort": null}
                    # with a 400.
                    effort = reasoning_cfg.get("effort") or "medium"
                    # Codex backend rejects "minimal"; clamp to "low" to
                    # match the main-agent Codex transport behavior.
                    if effort == "minimal":
                        effort = "low"
                    resp_kwargs["reasoning"] = {
                        "effort": effort,
                        "summary": "auto",
                    }
                    resp_kwargs["include"] = ["reasoning.encrypted_content"]

        # Tools support for auxiliary callers (e.g. skills_hub) that pass function schemas
        tools = kwargs.get("tools")
        if tools:
            # Deep-copy before sanitizing — ``list(tools)`` is only a shallow
            # copy of the outer list, but the sanitizers mutate the inner
            # parameter dicts in place.  Without a deep copy the caller's
            # tool registry permanently loses its slash-containing enum
            # constraints after the first auxiliary call.
            try:
                import copy as _copy
                from tools.schema_sanitizer import (
                    strip_pattern_and_format,
                    strip_slash_enum,
                )
                tools = _copy.deepcopy(list(tools))
                tools, _ = strip_pattern_and_format(tools)
                tools, _ = strip_slash_enum(tools)
            except Exception as exc:
                logger.warning(
                    "Auxiliary client: failed to sanitize tool schemas for "
                    "Codex Responses path: %s", exc,
                )
            converted = []
            for t in tools:
                fn = t.get("function", {}) if isinstance(t, dict) else {}
                name = fn.get("name")
                if not name:
                    continue
                converted.append({
                    "type": "function",
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            if converted:
                resp_kwargs["tools"] = converted

        # Stable prompt-cache routing for the Codex/Responses aux path, mirroring
        # the main transport (agent/transports/codex.py::build_kwargs, which sets
        # prompt_cache_key = _content_cache_key(instructions, tools)). Without
        # this, MoA acting-aggregator and other auxiliary Responses calls stay
        # cache-cold while the main Responses transport is warm.
        # The key is content-addressed from the static prefix (instructions +
        # tool schemas) so it stays warm across turns/fires. Guard the top-level
        try:
            from agent.transports.codex import (
                _cache_scope_from_session_id,
                _content_cache_key,
            )
            from utils import base_url_host_matches

            if "prompt_cache_key" not in resp_kwargs:
                # Scope by the owning turn's conversation so two unrelated
                # sessions with the same instructions/tools (e.g. compression,
                # MoA, flush_memories firing back-to-back on different
                # sessions) don't bucket-share a prompt cache slot.
                # Prefer the rotation-stable logical scope threaded through
                # set_runtime_main (compression-lineage root,) and
                # fall back to the physical session id, mirroring the main
                # transport (agent/transports/codex.py::build_kwargs).
                _scope = _cache_scope_from_session_id(
                    _runtime_main_value("cache_scope")
                    or _runtime_main_value("session_id")
                )
                _cache_key = _content_cache_key(instructions, resp_kwargs.get("tools"), _scope)
                if _cache_key:
                    resp_kwargs["prompt_cache_key"] = _cache_key
        except Exception:
            logger.debug(
                "Codex auxiliary: prompt_cache_key derivation skipped", exc_info=True
            )

        # Stream and collect the response
        text_parts: List[str] = []
        tool_calls_raw: List[Any] = []
        usage = None
        total_timeout = timeout if isinstance(timeout, (int, float)) and timeout > 0 else None
        deadline = time.monotonic() + float(total_timeout) if total_timeout else None
        timed_out = threading.Event()
        timeout_timer: Optional[threading.Timer] = None
        # A protected provider call may outlive its owning compression attempt:
        # the owner returns promptly on hard cancellation while this adapter is
        # still blocked in the SDK stream on its isolated worker. Timer threads
        # do not inherit this worker's thread-local protection state, so freeze
        # the hard-cancel source here, before creating the timer.
        protected_cancel_check = (
            _capture_aux_cancel_check() if _aux_interrupt_protected() else None
        )
        attempt_stream_lock = threading.Lock()
        attempt_stream: List[Any] = []

        def _timeout_message() -> str:
            return f"Codex auxiliary Responses stream exceeded {float(total_timeout):.1f}s total timeout"

        def _close_client_on_timeout() -> None:
            begin_timeout_cleanup = getattr(
                protected_cancel_check, "begin_timeout_cleanup", None
            )
            if callable(begin_timeout_cleanup):
                timeout_won = bool(begin_timeout_cleanup())
            else:
                timeout_won = not (
                    callable(protected_cancel_check)
                    and _captured_aux_cancel_requested(protected_cancel_check)
                )
            # Publish transport timeout only after the attempt-local decision is
            # fixed, so owner polling cannot observe completion in between.
            timed_out.set()
            if not timeout_won:
                # The request owner already hard-cancelled this attempt. The
                # OpenAI client is process-shared, so closing/evicting it here
                # would disrupt unrelated sessions. Wake only this attempt's
                # event stream when responses.create() returned one in time;
                # otherwise rely on the bounded SDK/provider timeout.
                with attempt_stream_lock:
                    stream = attempt_stream[0] if attempt_stream else None
                close_stream = getattr(stream, "close", None)
                if callable(close_stream):
                    try:
                        close_stream()
                    except Exception:
                        logger.debug(
                            "Codex auxiliary: cancelled attempt stream close "
                            "during timeout failed",
                            exc_info=True,
                        )
                return
            close = getattr(self._client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Codex auxiliary: client close during timeout failed", exc_info=True)
            # The cached auxiliary client wraps this same ``self._client``
            # (or *is* a ``CodexAuxiliaryClient`` whose ``_real_client`` is
            # this instance).  After we close the httpx transport above, the
            # cache must drop that entry — otherwise the next auxiliary call
            # (compression retry, memory flush, etc.) reuses the dead client
            # and fails fast with a connection error. See.
            try:
                _evict_cached_client_instance(self._client)
            except Exception:
                logger.debug("Codex auxiliary: cache eviction on timeout failed", exc_info=True)

        def _check_cancelled() -> None:
            if deadline is not None and time.monotonic() >= deadline:
                if not timed_out.is_set():
                    _close_client_on_timeout()
                raise TimeoutError(_timeout_message())
            try:
                from tools.interrupt import is_interrupted
                # Honor interrupt protection for atomic aux tasks (compression):
                # a mid-flight gateway interrupt must NOT abort the summary call
                # and trigger a degraded fallback marker. Explicit host
                # cancellation has its own frozen exception; timeouts above still
                # fire and other aux tasks remain interruptible.
                if _aux_interrupt_cancel_requested():
                    raise AuxiliaryExplicitCancellation()
                if is_interrupted() and not _aux_interrupt_protected():
                    raise InterruptedError("Codex auxiliary Responses stream interrupted")
            except (InterruptedError, AuxiliaryExplicitCancellation):
                raise
            except Exception:
                # Interrupt state is a best-effort UX hook; never make it a
                # new failure mode for auxiliary calls.
                pass

        try:
            if total_timeout:
                timeout_timer = threading.Timer(float(total_timeout), _close_client_on_timeout)
                timeout_timer.daemon = True
                timeout_timer.start()
            _check_cancelled()

            # Event-driven Responses streaming via the low-level
            # ``responses.create(stream=True)`` path.  The high-level
            # ``responses.stream(...)`` helper does post-hoc typed
            # reconstruction from ``response.completed.response.output``,
            # which the chatgpt.com Codex backend has been observed to
            # return as ``null`` (gpt-5.5, May 2026) — that crashes the SDK
            # with ``TypeError: 'NoneType' object is not iterable``.
            # Consuming raw events and assembling the final response
            # ourselves from ``response.output_item.done`` makes us
            # structurally immune to that drift.
            from agent.codex_runtime import _consume_codex_event_stream

            stream_kwargs = dict(resp_kwargs)
            stream_kwargs["stream"] = True

            def _on_each_event(_event: Any) -> None:
                # Re-check timeout/cancellation per event, matching the
                # cadence the old in-line ``_check_cancelled()`` used.
                # Each SSE event is also forward progress for hosts watching
                # a progress hook (gateway session hygiene): a reasoning
                # model streaming a long summary must not look hung.
                _notify_aux_progress()
                _check_cancelled()

            event_stream = self._client.responses.create(**stream_kwargs)
            with attempt_stream_lock:
                attempt_stream.append(event_stream)
            # The timer can fire while responses.create() is blocked. If the
            # cancelled attempt had no stream to close at that instant, close it
            # now that it is safely attempt-owned; never touch the shared client.
            if (
                timed_out.is_set()
                and callable(protected_cancel_check)
                and _captured_aux_cancel_requested(protected_cancel_check)
            ):
                close_fn = getattr(event_stream, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        logger.debug(
                            "Codex auxiliary: late cancelled attempt stream close failed",
                            exc_info=True,
                        )
            try:
                # Some Codex-compatible hosts accept ``stream=True`` but return
                # a completed Responses object instead of an SSE iterator. Do
                # not hand that object to the event consumer: typed Responses
                # (and compatibility shims such as SimpleNamespace) are not
                # event streams and may not be iterable at all.
                if hasattr(event_stream, "output"):
                    final = event_stream
                else:
                    final = _consume_codex_event_stream(
                        event_stream,
                        model=str(resp_kwargs.get("model") or model),
                        on_event=_on_each_event,
                    )
            finally:
                close_fn = getattr(event_stream, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
                with attempt_stream_lock:
                    attempt_stream.clear()

            if final is None:
                raise RuntimeError("Codex auxiliary Responses stream did not return a final response")

            # Extract text and tool calls from the Responses output.
            # Items may be SimpleNamespace (raw-event path) or dicts
            # (some legacy fallback paths), so handle both shapes.
            def _item_get(obj: Any, key: str, default: Any = None) -> Any:
                val = getattr(obj, key, None)
                if val is None and isinstance(obj, dict):
                    val = obj.get(key, default)
                return val if val is not None else default

            for item in (getattr(final, "output", None) or []):
                item_type = _item_get(item, "type")
                if item_type == "message":
                    for part in (_item_get(item, "content") or []):
                        ptype = _item_get(part, "type")
                        if ptype in {"output_text", "text"}:
                            text_parts.append(_item_get(part, "text", ""))
                elif item_type == "function_call":
                    tool_calls_raw.append(SimpleNamespace(
                        id=_item_get(item, "call_id", ""),
                        type="function",
                        function=SimpleNamespace(
                            name=_item_get(item, "name", ""),
                            arguments=_item_get(item, "arguments", "{}"),
                        ),
                    ))

            resp_usage = getattr(final, "usage", None)
            if resp_usage:
                usage = SimpleNamespace(
                    prompt_tokens=getattr(resp_usage, "input_tokens", 0)
                        or (resp_usage.get("input_tokens", 0) if isinstance(resp_usage, dict) else 0),
                    completion_tokens=getattr(resp_usage, "output_tokens", 0)
                        or (resp_usage.get("output_tokens", 0) if isinstance(resp_usage, dict) else 0),
                    total_tokens=getattr(resp_usage, "total_tokens", 0)
                        or (resp_usage.get("total_tokens", 0) if isinstance(resp_usage, dict) else 0),
                )
        except Exception as exc:
            if timed_out.is_set():
                raise TimeoutError(_timeout_message()) from exc
            logger.debug("Codex auxiliary Responses API call failed: %s", exc)
            raise
        finally:
            if timeout_timer is not None:
                timeout_timer.cancel()

        content = "".join(text_parts).strip() or None

        # Build a response that looks like chat.completions
        message = SimpleNamespace(
            role="assistant",
            content=content,
            tool_calls=tool_calls_raw or None,
        )
        choice = SimpleNamespace(
            index=0,
            message=message,
            finish_reason="stop" if not tool_calls_raw else "tool_calls",
        )
        return SimpleNamespace(
            choices=[choice],
            model=model,
            usage=usage,
        )


class _CodexChatShim:
    """Wraps the adapter to provide client.chat.completions.create()."""

    def __init__(self, adapter: _CodexCompletionsAdapter):
        self.completions = adapter


class CodexAuxiliaryClient:
    """OpenAI-client-compatible wrapper that routes through Codex Responses API.

    Consumers can call client.chat.completions.create(**kwargs) as normal.
    Also exposes .api_key and .base_url for introspection by async wrappers.
    """

    def __init__(self, real_client: OpenAI, model: str):
        self._real_client = real_client
        adapter = _CodexCompletionsAdapter(real_client, model)
        self.chat = _CodexChatShim(adapter)
        self.api_key = real_client.api_key
        self.base_url = real_client.base_url

    def close(self):
        self._real_client.close()


class _AsyncCodexCompletionsAdapter:
    """Async version of the Codex Responses adapter.

    Wraps the sync adapter via asyncio.to_thread() so async consumers
    (web_tools, session_search) can await it as normal.
    """

    def __init__(self, sync_adapter: _CodexCompletionsAdapter):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncCodexChatShim:
    def __init__(self, adapter: _AsyncCodexCompletionsAdapter):
        self.completions = adapter


class AsyncCodexAuxiliaryClient:
    """Async-compatible wrapper matching AsyncOpenAI.chat.completions.create()."""

    def __init__(self, sync_wrapper: "CodexAuxiliaryClient"):
        sync_adapter = sync_wrapper.chat.completions
        async_adapter = _AsyncCodexCompletionsAdapter(sync_adapter)
        self.chat = _AsyncCodexChatShim(async_adapter)
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url
        # Mirror the sync wrapper's _real_client so cache eviction by leaf
        # OpenAI client (e.g. _close_client_on_timeout in) drops
        # this async entry too. Without this, sync and async cache entries
        # diverge on poisoning: the sync entry is evicted but the async
        # entry keeps reusing the closed transport, failing every
        # subsequent async aux call with 'Connection error' until the
        # gateway restarts.
        self._real_client = sync_wrapper._real_client




def _read_codex_access_token() -> Optional[str]:
    """Read a valid, non-expired Codex OAuth access token from Pilotage auth store.

    If a credential pool exists but currently has no selectable runtime entry
    (for example all pool slots are marked exhausted), fall back to the
    profile's auth.json token instead of hard-failing. This keeps explicit
    fallback-to-Codex working when the pool state is stale but the stored OAuth
    token is still valid.
    """
    pool_present, entry = _select_pool_entry("openai-codex")
    if pool_present:
        token = _pool_runtime_api_key(entry)
        if token:
            return token

    try:
        from pilotage_cli.auth import _read_codex_tokens
        data = _read_codex_tokens()
        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            return None

        # Check JWT expiry — expired tokens block the auto chain.
        try:
            import base64
            payload = access_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            exp = claims.get("exp", 0)
            if exp and time.time() > exp:
                logger.debug("Codex access token expired (exp=%s), skipping", exp)
                return None
        except Exception:
            pass  # Non-JWT token or decode error — use as-is

        return access_token.strip()
    except Exception as exc:
        logger.debug("Could not read Codex auth for auxiliary client: %s", exc)
        return None


def _resolve_api_key_provider() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Try each API-key provider in PROVIDER_REGISTRY order.

    Returns (client, model) for the first provider with usable runtime
    credentials, or (None, None) if none are configured.
    """
    try:
        from pilotage_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials
    except ImportError:
        logger.debug("Could not import PROVIDER_REGISTRY for API-key fallback")
        return None, None

    for provider_id, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != "api_key":
            continue
        pool_present, entry = _select_pool_entry(provider_id)
        if pool_present:
            api_key = _pool_runtime_api_key(entry)
            if not api_key:
                continue

            raw_base_url = _pool_runtime_base_url(entry, pconfig.inference_base_url) or pconfig.inference_base_url
            base_url = _to_openai_base_url(raw_base_url)
            model = _get_aux_model_for_provider(provider_id) or None
            if model is None:
                continue  # skip provider if we don't know a valid aux model
            logger.debug("Auxiliary text client: %s (%s) via pool", pconfig.name, model)
            extra = {}
            try:
                from providers import get_provider_profile as _gpf_aux
                _ph_aux = _gpf_aux(provider_id)
                if _ph_aux and _ph_aux.default_headers:
                    extra["default_headers"] = dict(_ph_aux.default_headers)
            except Exception:
                pass
            _merged_aux = _apply_user_default_headers(extra.get("default_headers"))
            if _merged_aux:
                extra["default_headers"] = _merged_aux
            return _create_openai_client(api_key=api_key, base_url=base_url, **extra), model

        creds = resolve_api_key_provider_credentials(provider_id)
        api_key = str(creds.get("api_key", "")).strip()
        if not api_key:
            continue

        raw_base_url = str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url
        base_url = _to_openai_base_url(raw_base_url)
        model = _get_aux_model_for_provider(provider_id) or None
        if model is None:
            continue  # skip provider if we don't know a valid aux model
        logger.debug("Auxiliary text client: %s (%s)", pconfig.name, model)
        extra = {}
        try:
            from providers import get_provider_profile as _gpf_aux2
            _ph_aux2 = _gpf_aux2(provider_id)
            if _ph_aux2 and _ph_aux2.default_headers:
                extra["default_headers"] = dict(_ph_aux2.default_headers)
        except Exception:
            pass
        _merged_aux2 = _apply_user_default_headers(extra.get("default_headers"))
        if _merged_aux2:
            extra["default_headers"] = _merged_aux2
        return _create_openai_client(api_key=api_key, base_url=base_url, **extra), model

    return None, None


# ── Provider resolution helpers ─────────────────────────────────────────────


_paid_lane_warned: set = set()


def _read_main_model() -> str:
    """Read the user's configured main model from config.yaml.

    config.yaml model.default is the single source of truth for the active
    model. Environment variables are no longer consulted.

    Runtime override: when an AIAgent is active with a CLI/gateway-provided
    model that differs from config.yaml, ``set_runtime_main()`` records the
    override in a process-local global. This is consulted FIRST so tools
    that gate on "the active main model" (e.g. ``vision_analyze``'s native
    fast path) see the live runtime, not the persisted config default.
    """
    override = _runtime_main_value("model")
    if isinstance(override, str) and override.strip():
        return override.strip()
    try:
        from pilotage_cli.config import load_config_readonly
        cfg = load_config_readonly()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, str) and model_cfg.strip():
            return model_cfg.strip()
        if isinstance(model_cfg, dict):
            default = model_cfg.get("default", "")
            if isinstance(default, str) and default.strip():
                return default.strip()
    except Exception:
        pass
    return ""


def _read_main_provider() -> str:
    """Read the user's configured main provider from config.yaml.

    Returns the lowercase provider id (e.g. "openai") or ""
    if not configured.

    Runtime override: see ``_read_main_model`` — same mechanism for the
    provider half of the runtime tuple.
    """
    override = _runtime_main_value("provider")
    if isinstance(override, str) and override.strip():
        return override.strip().lower()
    try:
        from pilotage_cli.config import load_config_readonly
        cfg = load_config_readonly()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            provider = model_cfg.get("provider", "")
            if isinstance(provider, str) and provider.strip():
                return provider.strip().lower()
    except Exception:
        pass
    return ""


def _read_main_api_key() -> str:
    """Read the user's main model API key from the runtime override or config.

    Mirrors ``_read_main_model`` / ``_read_main_provider``: checks the
    process-local ``_RUNTIME_MAIN_API_KEY`` override first (set by
    ``set_runtime_main`` when an AIAgent is active), then falls back to
    ``model.api_key`` in config.yaml.

    Used by the ``custom`` provider fallback chain so that auxiliary tasks
    configured with an explicit ``base_url`` but empty ``api_key`` inherit
    the main model's credentials instead of falling to ``no-key-required``
    .
    """
    override = _runtime_main_value("api_key")
    if isinstance(override, str) and override.strip():
        return override.strip()
    try:
        from pilotage_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            key = model_cfg.get("api_key", "")
            if isinstance(key, str) and key.strip():
                return key.strip()
    except Exception:
        pass
    return ""


def _read_main_base_url() -> str:
    """Read the main model's base_url from the runtime override or config.

    Same override-then-config pattern as ``_read_main_api_key``.
    """
    override = _runtime_main_value("base_url")
    if isinstance(override, str) and override.strip():
        return override.strip()
    try:
        from pilotage_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            base = model_cfg.get("base_url", "")
            if isinstance(base, str) and base.strip():
                return base.strip()
    except Exception:
        pass
    return ""


def _read_main_model_for_aux() -> str:
    """Main model, for aux fallback chains that pre-fill a missing model."""
    return _read_main_model()


def _read_main_api_key_if_same_host(aux_base_url: str) -> str:
    """Return the main api_key only when *aux_base_url* points at the same
    host as the main model's base_url.

    The use case is an auxiliary task sharing the main model's
    self-hosted gateway (same host, different model) with an empty per-task
    api_key. Inheriting unconditionally would send the main credential to
    ANY host a misconfigured aux base_url names — a cross-host credential
    leak. A host mismatch keeps the previous fail-safe behavior
    (``no-key-required`` → 401).
    """
    aux_host = base_url_hostname(aux_base_url)
    if not aux_host:
        return ""
    main_host = base_url_hostname(_read_main_base_url())
    if not main_host or aux_host != main_host:
        return ""
    return _read_main_api_key()


# Compatibility mirrors for older readers/tests. The authoritative value is
# the ContextVar below: gateway sessions can overlap in one process, so a
# process-global tuple is not safe as routing or cache-key input.
_RUNTIME_MAIN_PROVIDER: str = ""
_RUNTIME_MAIN_MODEL: str = ""
_RUNTIME_MAIN_BASE_URL: str = ""
_RUNTIME_MAIN_API_KEY: Any = ""
_RUNTIME_MAIN_API_MODE: str = ""
_RUNTIME_MAIN_AUTH_MODE: str = ""
_RUNTIME_MAIN_CONTEXT: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("auxiliary_runtime_main", default=None)
)

_RELAY_AUX_CALL_CONTEXT: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("auxiliary_relay_call", default=None)
)


def _relay_auxiliary_call(callback):
    """Give every physical retry in one auxiliary call a shared Relay identity."""

    @functools.wraps(callback)
    def wrapped(*args, **kwargs):
        task = args[0] if args else kwargs.get("task")
        token = _RELAY_AUX_CALL_CONTEXT.set({
            "task": str(task or "unknown"),
            "request_id": f"aux-{uuid.uuid4().hex}",
            "attempt_count": 0,
            "provider": "",
            "model": "",
            "response_model": None,
            "api_mode": "chat_completions",
        })
        try:
            return callback(*args, **kwargs)
        except BaseException:
            _fail_relay_auxiliary_call()
            raise
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)

    return wrapped


def _relay_auxiliary_call_async(callback):
    """Async counterpart to :func:`_relay_auxiliary_call`."""

    @functools.wraps(callback)
    async def wrapped(*args, **kwargs):
        task = args[0] if args else kwargs.get("task")
        token = _RELAY_AUX_CALL_CONTEXT.set({
            "task": str(task or "unknown"),
            "request_id": f"aux-{uuid.uuid4().hex}",
            "attempt_count": 0,
            "provider": "",
            "model": "",
            "response_model": None,
            "api_mode": "chat_completions",
        })
        try:
            return await callback(*args, **kwargs)
        except BaseException:
            _fail_relay_auxiliary_call()
            raise
        finally:
            _RELAY_AUX_CALL_CONTEXT.reset(token)

    return wrapped


def _set_relay_auxiliary_route(
    provider: str | None,
    model: str | None,
    api_mode: str | None,
) -> None:
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return
    context["provider"] = str(provider or "auxiliary")
    context["model"] = str(model or "unknown")
    context["response_model"] = None
    context["api_mode"] = str(api_mode or "chat_completions")


def _record_route_info(
    route_info: Optional[Dict[str, str]],
    provider: Optional[str],
    model: Optional[str],
) -> None:
    """Expose the concrete route selected for one auxiliary call."""
    if route_info is not None:
        route_info["provider"] = provider or "auto"
        route_info["model"] = model or "default"


def _relay_auxiliary_metadata(
    *,
    provider: str | None = None,
    api_mode: str | None = None,
) -> tuple[str, str, dict[str, Any]] | None:
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return None
    attempt_count = int(context.get("attempt_count") or 0)
    context["attempt_count"] = attempt_count + 1
    provider_name = str(provider or context.get("provider") or "auxiliary")
    model_name = str(context.get("model") or "unknown")
    return provider_name, model_name, {
        "api_mode": str(api_mode or context.get("api_mode") or "chat_completions"),
        "api_request_id": str(context["request_id"]),
        "call_role": f"auxiliary:{context['task']}",
        "retry_count": attempt_count,
        "auxiliary_task": str(context["task"]),
    }


def _relay_sync_completion(
    client: Any,
    kwargs: dict[str, Any],
    *,
    provider: str | None = None,
    api_mode: str | None = None,
    create: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    callback = create or (lambda request: client.chat.completions.create(**request))
    route = _relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    # Protected compression calls isolate only the provider callback and stream
    # aggregation.  The owning thread remains free to unwind its lease/DB
    # transaction on hard cancel without touching the process-shared client.
    if route is None:
        return _run_protected_sync_provider_call(callback, kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm

    return relay_llm.execute_current(
        kwargs,
        lambda request: _run_protected_sync_provider_call(callback, request),
        name=provider_name,
        model_name=str(kwargs.get("model") or fallback_model),
        metadata=metadata,
        defer_logical_completion=True,
    )


async def _relay_async_completion(
    client: Any,
    kwargs: dict[str, Any],
    *,
    provider: str | None = None,
    api_mode: str | None = None,
    create: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    callback = create or (lambda request: client.chat.completions.create(**request))
    route = _relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    if route is None:
        return await callback(kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm

    return await relay_llm.execute_current_async(
        kwargs,
        callback,
        name=provider_name,
        model_name=str(kwargs.get("model") or fallback_model),
        metadata=metadata,
        defer_logical_completion=True,
    )


def _relay_sync_stream(
    client: Any,
    kwargs: dict[str, Any],
    *,
    provider: str | None = None,
    api_mode: str | None = None,
) -> Any:
    route = _relay_auxiliary_metadata(provider=provider, api_mode=api_mode)
    if route is None:
        return client.chat.completions.create(**kwargs)
    provider_name, fallback_model, metadata = route
    from agent import relay_llm

    return relay_llm.stream_current(
        kwargs,
        lambda request: client.chat.completions.create(**request),
        name=provider_name,
        model_name=str(kwargs.get("model") or fallback_model),
        finalizer=dict,
        metadata=metadata,
        completed_response_predicate=lambda value: hasattr(value, "choices"),
    )
_RUNTIME_MAIN_COMPAT_SNAPSHOT: Tuple[Any, ...] = ("", "", "", "", "", "")
_RUNTIME_MAIN_COMPAT_LOCK = threading.Lock()


def _compat_runtime_main() -> Optional[Dict[str, Any]]:
    """Expose deliberately patched legacy globals in a single main context.

    ``set_runtime_main`` mirrors values into the old module attributes for
    introspection, but those mirrors must never become runtime inputs. A direct
    patch is recognized only when it differs from the mirrored snapshot and
    only on the main thread, keeping concurrent session workers isolated.
    """
    if threading.current_thread() is not threading.main_thread():
        return None
    values = (
        _RUNTIME_MAIN_PROVIDER,
        _RUNTIME_MAIN_MODEL,
        _RUNTIME_MAIN_BASE_URL,
        _RUNTIME_MAIN_API_KEY,
        _RUNTIME_MAIN_API_MODE,
        _RUNTIME_MAIN_AUTH_MODE,
    )
    if values == _RUNTIME_MAIN_COMPAT_SNAPSHOT:
        return None
    return dict(zip(_MAIN_RUNTIME_FIELDS, values))


def _runtime_main_value(field: str) -> Any:
    """Read one runtime field through context-local/controlled legacy state."""
    runtime = _RUNTIME_MAIN_CONTEXT.get()
    if runtime is None:
        runtime = _compat_runtime_main()
    if isinstance(runtime, dict):
        value = runtime.get(field)
        if value:
            return value
    return ""


def set_runtime_main(
    provider: str,
    model: str,
    *,
    requested_provider: str = "",
    base_url: str = "",
    api_key: Any = "",
    api_mode: str = "",
    auth_mode: str = "",
    session_id: str = "",
    cache_scope: str = "",
) -> contextvars.Token:
    """Record the current context's live main runtime for auxiliary routing.

    Context-local state prevents concurrent gateway sessions from overwriting
    one another while retaining compatibility mirrors for legacy readers.

    ``cache_scope`` is the rotation-stable logical cache scope (compression-
    lineage root — agent/prompt_cache_scope.py) resolved once per turn by
    turn_context; auxiliary Responses calls prefer it over ``session_id``
    for prompt_cache_key derivation.
    """
    global _RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL
    global _RUNTIME_MAIN_BASE_URL, _RUNTIME_MAIN_API_KEY, _RUNTIME_MAIN_API_MODE
    global _RUNTIME_MAIN_AUTH_MODE, _RUNTIME_MAIN_COMPAT_SNAPSHOT
    runtime = {
        "provider": (provider or "").strip().lower(),
        "requested_provider": (requested_provider or "").strip().lower(),
        "model": (model or "").strip(),
        "base_url": (base_url or "").strip(),
        "api_key": (
            api_key.strip()
            if isinstance(api_key, str)
            else api_key if callable(api_key) else ""
        ),
        "api_mode": (api_mode or "").strip(),
        "auth_mode": (auth_mode or "").strip().lower(),
        "session_id": (session_id or "").strip(),
        "cache_scope": (cache_scope or "").strip(),
    }
    # Publish authoritative context before updating locked compatibility
    # mirrors; concurrent sessions never read those mirrors at runtime.
    token = _RUNTIME_MAIN_CONTEXT.set(runtime)
    with _RUNTIME_MAIN_COMPAT_LOCK:
        (
            _RUNTIME_MAIN_PROVIDER,
            _RUNTIME_MAIN_MODEL,
            _RUNTIME_MAIN_BASE_URL,
            _RUNTIME_MAIN_API_KEY,
            _RUNTIME_MAIN_API_MODE,
            _RUNTIME_MAIN_AUTH_MODE,
        ) = (runtime[field] for field in _MAIN_RUNTIME_FIELDS)
        _RUNTIME_MAIN_COMPAT_SNAPSHOT = tuple(
            runtime[field] for field in _MAIN_RUNTIME_FIELDS
        )
    return token


def reset_runtime_main(token: contextvars.Token) -> None:
    """Restore the runtime binding that preceded one scoped turn."""
    if token is None:
        return
    try:
        _RUNTIME_MAIN_CONTEXT.reset(token)
    except (RuntimeError, ValueError):
        # A token cannot be reset from another copied Context. Background
        # workers inherit values, not ownership of the parent's token.
        pass


@contextlib.contextmanager
def scoped_runtime_main(main_runtime: Optional[Dict[str, Any]]):
    """Temporarily bind an explicit runtime without touching legacy mirrors."""
    runtime = _normalize_main_runtime(main_runtime)
    token = _RUNTIME_MAIN_CONTEXT.set(runtime or None)
    try:
        yield runtime
    finally:
        _RUNTIME_MAIN_CONTEXT.reset(token)


def clear_runtime_main() -> None:
    """Clear the runtime override in the current context."""
    global _RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL
    global _RUNTIME_MAIN_BASE_URL, _RUNTIME_MAIN_API_KEY, _RUNTIME_MAIN_API_MODE
    global _RUNTIME_MAIN_AUTH_MODE, _RUNTIME_MAIN_COMPAT_SNAPSHOT
    _RUNTIME_MAIN_CONTEXT.set(None)
    with _RUNTIME_MAIN_COMPAT_LOCK:
        _RUNTIME_MAIN_PROVIDER = ""
        _RUNTIME_MAIN_MODEL = ""
        _RUNTIME_MAIN_BASE_URL = ""
        _RUNTIME_MAIN_API_KEY = ""
        _RUNTIME_MAIN_API_MODE = ""
        _RUNTIME_MAIN_AUTH_MODE = ""
        _RUNTIME_MAIN_COMPAT_SNAPSHOT = ("", "", "", "", "", "")


def _resolve_custom_runtime() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve the active custom/main endpoint the same way the main CLI does.

    This covers both env-driven OPENAI_BASE_URL setups and config-saved custom
    endpoints where the base URL lives in config.yaml instead of the live
    environment.
    """
    try:
        from pilotage_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="custom")
    except Exception as exc:
        logger.debug("Auxiliary client: custom runtime resolution failed: %s", exc)
        runtime = None

    if not isinstance(runtime, dict):
        openai_base = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
        openai_key = _scoped_key_env("OPENAI_API_KEY")
        if not openai_base:
            return None, None, None
        runtime = {
            "base_url": openai_base,
            "api_key": openai_key,
        }

    custom_base = runtime.get("base_url")
    custom_key = runtime.get("api_key")
    custom_mode = runtime.get("api_mode")
    if not isinstance(custom_base, str) or not custom_base.strip():
        return None, None, None

    custom_base = custom_base.strip().rstrip("/")
    # Local servers don't require auth.
    # Use a placeholder key — the OpenAI SDK requires a non-empty string but
    # local servers ignore the Authorization header.  Same fix as cli.py
    # _ensure_runtime_credentials.
    if not isinstance(custom_key, str) or not custom_key.strip():
        custom_key = "no-key-required"

    if not isinstance(custom_mode, str) or not custom_mode.strip():
        custom_mode = None

    return custom_base, custom_key.strip(), custom_mode


def _current_custom_base_url() -> str:
    custom_base, _, _ = _resolve_custom_runtime()
    return custom_base or ""


def _validate_proxy_env_urls() -> None:
    """Fail fast with a clear error when proxy env vars have malformed URLs.

    Common cause: shell config (e.g. .zshrc) with a typo like
    ``export HTTP_PROXY=http://127.0.0.1:6153export NEXT_VAR=...``
    which concatenates 'export' into the port number.  Without this
    check the OpenAI/httpx client raises a cryptic ``Invalid port``
    error that doesn't name the offending env var.
    """
    from urllib.parse import urlparse

    normalize_proxy_env_vars()

    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        value = str(os.environ.get(key) or "").strip()
        if not value:
            continue
        try:
            parsed = urlparse(value)
            if parsed.scheme:
                _ = parsed.port          # raises ValueError for e.g. '6153export'
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed proxy environment variable {key}={value!r}. "
                "Fix or unset your proxy settings and try again."
            ) from exc


def _validate_base_url(base_url: str) -> None:
    """Reject obviously broken custom endpoint URLs before they reach httpx."""
    from urllib.parse import urlparse

    candidate = str(base_url or "").strip()
    if not candidate or candidate.startswith("acp://"):
        return
    try:
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            _ = parsed.port              # raises ValueError for malformed ports
    except ValueError as exc:
        raise RuntimeError(
            f"Malformed custom endpoint URL: {candidate!r}. "
            "Run `pilotage setup` or `pilotage model` and enter a valid http(s) base URL."
        ) from exc


def _try_custom_endpoint() -> Tuple[Optional[Any], Optional[str]]:
    runtime = _resolve_custom_runtime()
    if len(runtime) == 2:
        custom_base, custom_key = runtime
        custom_mode = None
    else:
        custom_base, custom_key, custom_mode = runtime
    if not custom_base or not custom_key:
        return None, None
    if custom_base.lower().startswith(_CODEX_AUX_BASE_URL.lower()):
        return None, None
    model = _read_main_model_for_aux() or "gpt-4o-mini"
    logger.debug("Auxiliary client: custom endpoint (%s, api_mode=%s)", model, custom_mode or "chat_completions")
    _clean_base, _dq = _extract_url_query_params(custom_base)
    _extra = {"default_query": _dq} if _dq else {}
    # User-configured model.default_headers override the SDK's identifying
    # headers (User-Agent: OpenAI/Python ..., X-Stainless-*) on this custom
    # endpoint's auxiliary calls too — matching the main agent client so the
    # whole session reaches a gateway/WAF that rejects the SDK fingerprint.
    _custom_headers = _apply_user_default_headers(None)
    if _custom_headers:
        _extra["default_headers"] = _custom_headers
    if custom_mode == "codex_responses":
        real_client = _create_openai_client(api_key=custom_key, base_url=_clean_base, **_extra)
        return CodexAuxiliaryClient(real_client, model), model
    return _create_openai_client(api_key=custom_key, base_url=_clean_base, **_extra), model


def _build_codex_client(model: str) -> Tuple[Optional[Any], Optional[str]]:
    """Build a CodexAuxiliaryClient for an explicitly-requested model.

    There is no auto-selection of the Codex model: the ChatGPT-account
    Codex endpoint's accepted model list is an undocumented, drifting
    allow-list, so any hardcoded default we pick goes stale.  The caller
    is responsible for passing the model (e.g. from the user's own
    ``model.model`` or ``auxiliary.<task>.model`` config).

    Returns (None, None) when no Codex OAuth token is available.
    """
    if not model:
        logger.warning(
            "Auxiliary client: openai-codex requested without a model; "
            "pass model explicitly (auxiliary.<task>.model in config.yaml)."
        )
        return None, None
    pool_present, entry = _select_pool_entry("openai-codex")
    if pool_present:
        codex_token = _pool_runtime_api_key(entry)
        if codex_token:
            base_url = _pool_runtime_base_url(entry, _CODEX_AUX_BASE_URL) or _CODEX_AUX_BASE_URL
        else:
            codex_token = _read_codex_access_token()
            if not codex_token:
                return None, None
            base_url = _CODEX_AUX_BASE_URL
    else:
        codex_token = _read_codex_access_token()
        if not codex_token:
            return None, None
        base_url = _CODEX_AUX_BASE_URL
    logger.debug("Auxiliary client: Codex OAuth (%s via Responses API)", model)
    real_client = _create_openai_client(
        api_key=codex_token,
        base_url=base_url,
        default_headers=_codex_cloudflare_headers(codex_token),
    )
    return CodexAuxiliaryClient(real_client, model), model


_MAIN_RUNTIME_FIELDS = ("provider", "model", "base_url", "api_key", "api_mode", "auth_mode")
_MAIN_RUNTIME_CONTEXT_FIELDS = _MAIN_RUNTIME_FIELDS + ("requested_provider",)


def _normalize_main_runtime(main_runtime: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a sanitized copy of a live main-runtime override.

    Most fields are stripped strings. ``api_key`` may legitimately be a
    zero-arg callable (Azure Foundry Entra ID token provider) — preserve
    those as-is so auxiliary clients inherit the same authentication
    surface as the main agent. The OpenAI SDK accepts ``Callable[[], str]``
    for ``api_key`` and calls it before every request.
    """
    if main_runtime is None:
        # Context-local state is inherited by tool worker wrappers while
        # remaining isolated across concurrent gateway sessions. Never fall
        # back to compatibility mirrors here: another session may have written
        # them most recently, which would leak its endpoint/key into this call.
        main_runtime = _RUNTIME_MAIN_CONTEXT.get()
        if main_runtime is None:
            main_runtime = _compat_runtime_main()
    if not isinstance(main_runtime, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for field in _MAIN_RUNTIME_CONTEXT_FIELDS:
        value = main_runtime.get(field)
        # Preserve a callable api_key (Entra ID bearer provider) unchanged.
        if field == "api_key" and callable(value) and not isinstance(value, str):
            normalized[field] = value
            continue
        if isinstance(value, str) and value.strip():
            normalized[field] = value.strip()
    for identity_field in ("provider", "requested_provider"):
        identity = normalized.get(identity_field)
        if isinstance(identity, str):
            normalized[identity_field] = identity.lower()
    return normalized


def _is_payment_error(exc: Exception) -> bool:
    """Detect payment/credit/quota exhaustion errors.

    Returns True for HTTP 402 (Payment Required) and for 429/other errors
    whose message indicates billing exhaustion or daily quota exhaustion
    rather than transient rate limiting.

    Daily token quota errors (e.g. Bedrock "Too many tokens per day",
    Vertex AI "quota exceeded") are functionally equivalent to credit
    exhaustion — the provider cannot serve the request until the quota
    resets — and should trigger the same provider-fallback logic.
    """
    status = getattr(exc, "status_code", None)
    if status == 402:
        return True
    err_lower = str(exc).lower()
    # Providers include "credits" or "afford" in 402 bodies, but sometimes
    # wrap them in 429 or other codes.  Daily quota exhaustion uses different
    # language but is semantically identical to credit exhaustion.
    if status in {402, 403, 404, 429, None}:
        if any(kw in err_lower for kw in (
            "credits", "insufficient funds",
            "can only afford", "billing",
            "payment required",
            "out of funds", "run out of funds",
            "balance_depleted", "no usable credits",
            "model_not_supported_on_free_tier",
            "not available on the free tier",
            "requires a subscription", "upgrade for access",
            "upgrade for higher limits", "reached your session usage limit",
            # Daily / monthly / weekly quota exhaustion keywords
            "quota exceeded", "quota_exceeded",
            "too many tokens per day", "daily limit",
            "tokens per day", "daily quota",
            "resource exhausted",  # Vertex AI / gRPC quota errors
            "weekly usage limit", "weekly limit",  # OpenCode Go weekly subscription cap
        )):
            return True
    return False


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect rate-limit errors that warrant provider fallback.

    Returns True for HTTP 429 errors whose message indicates rate limiting
    (as opposed to billing/quota exhaustion, which _is_payment_error handles).
    Also catches OpenAI SDK RateLimitError instances that may not set
    .status_code on the exception object.
    """
    status = getattr(exc, "status_code", None)
    err_lower = str(exc).lower()

    # OpenAI SDK's RateLimitError sometimes omits .status_code —
    # detect by class name so we don't miss these. ( pattern)
    if type(exc).__name__ == "RateLimitError":
        return True

    if status == 429:
        # Distinguish rate-limit from billing: billing keywords are handled
        # by _is_payment_error, everything else on 429 is a rate limit.
        if any(kw in err_lower for kw in (
            "rate limit", "rate_limit", "too many requests",
            "try again", "retry after", "resets in",
        )):
            return True
        # Generic 429 without billing keywords = likely a rate limit
        if not any(kw in err_lower for kw in (
            "credits", "insufficient funds", "billing",
            "payment required", "can only afford",
            "out of funds", "run out of funds",
            "balance_depleted", "no usable credits",
            "model_not_supported_on_free_tier",
            "not available on the free tier",
        )):
            return True
    return False


def _is_timeout_error(exc: Exception) -> bool:
    """Detect a request timeout — the full-budget stall, distinct from a fast
    connection drop.

    A timeout burns the entire configured ``timeout`` before surfacing, so a
    same-provider retry on the critical compression path doubles the
    user-visible wall time. A streaming-close / dropped
    connection, by contrast, fails fast and is cheap to retry — those stay on
    the retry path even for compression.
    """
    try:
        from openai import APITimeoutError
        if isinstance(exc, APITimeoutError):
            return True
    except ImportError:
        pass
    if "Timeout" in type(exc).__name__:
        return True
    return "timed out" in str(exc).lower()


def _is_connection_error(exc: Exception) -> bool:
    """Detect connection/network errors that warrant provider fallback.

    Returns True for errors indicating the provider endpoint is unreachable
    (DNS failure, connection refused, TLS errors, timeouts).  These are
    distinct from API errors (4xx/5xx) which indicate the provider IS
    reachable but returned an error.
    """
    try:
        from openai import APIConnectionError, APITimeoutError
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
    except ImportError:
        pass
    # urllib3 / httpx / httpcore connection errors
    err_type = type(exc).__name__
    if any(kw in err_type for kw in ("Connection", "Timeout", "DNS", "SSL")):
        return True
    err_lower = str(exc).lower()
    if any(kw in err_lower for kw in (
        "connection refused", "name or service not known",
        "no route to host", "network is unreachable",
        "timed out", "connection reset",
        # httpcore / httpx streaming premature-close errors.  These surface
        # when a proxy or provider drops the connection mid-stream and are
        # transient by nature — the request should be retried or rerouted.
        # See.
        "incomplete chunked read",
        "peer closed connection",
        "response ended prematurely",
        "unexpected eof",
        "remoteprotocolerror",
        "localprotocolerror",
    )):
        return True
    return False


def _is_transient_transport_error(exc: Exception) -> bool:
    """Return True for a one-off transport blip worth retrying ON the
    same provider before any provider/model fallback.

    Covers connection/streaming-close errors (via the canonical
    ``_is_connection_error`` detector, shared so the two cannot drift) plus a
    pure 5xx/408 HTTP status. Deliberately narrow: this is the "retry the
    same target once" gate, distinct from ``_is_payment_error`` /
    ``_is_auth_error`` / ``_is_rate_limit_error`` which the except-chain
    handles by switching provider, refreshing creds, or rotating the pool.
    """
    if _is_connection_error(exc):
        return True
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return isinstance(status, int) and (status == 408 or 500 <= status < 600)


_DEFAULT_TRANSIENT_RETRIES = 2
# Base for exponential backoff between transient retries (seconds). Overridable
# so tests can zero it out and not sleep real wall-clock time.
_TRANSIENT_RETRY_BACKOFF_BASE = 1.0


def _transient_retry_count() -> int:
    """Number of same-provider retries for a transient transport blip.

    Read from ``auxiliary.transient_retries`` in config.yaml (default 2 →
    3 total attempts). Clamped to [0, 6] to bound worst-case wall time. A
    connection blip to a pinned auxiliary target (e.g. a MoA reference
    advisor) has no meaningful provider fallback, so a couple of retries with
    backoff is the difference between recovering and silently losing the call.
    Best-effort: any config-read failure falls back to the default.
    """
    try:
        from pilotage_cli.config import cfg_get, load_config

        val = cfg_get(load_config(), "auxiliary", "transient_retries")
        if val is None:
            return _DEFAULT_TRANSIENT_RETRIES
        n = int(val)
        return max(0, min(n, 6))
    except Exception:
        return _DEFAULT_TRANSIENT_RETRIES


def _is_auth_error(exc: Exception) -> bool:
    """Detect auth failures that should trigger provider-specific refresh."""
    status = getattr(exc, "status_code", None)
    if status == 401:
        return True
    err_lower = str(exc).lower()
    if "error code: 401" in err_lower or "authenticationerror" in type(exc).__name__.lower():
        return True
    return False


def _is_unsupported_parameter_error(exc: Exception, param: str) -> bool:
    """Detect provider 400s for an unsupported request parameter.

    Different OpenAI-compatible endpoints phrase the same class of error a few
    ways: ``Unsupported parameter: X``, ``unsupported_parameter`` with a
    ``param`` field, ``X is not supported``, ``unknown parameter: X``,
    ``unrecognized request argument: X``.  We match on both the parameter
    name and a generic "unsupported/unknown/unrecognized parameter" marker so
    call sites can reactively retry without the offending key instead of
    surfacing a noisy auxiliary failure.

    Generalizes the temperature-specific detector that originally shipped
    with so the same retry strategy can cover ``max_tokens``,
    ``seed``, ``top_p``, and any future quirk. Credit @nicholasrae 
    for the generalization pattern.
    """
    param_lower = (param or "").lower()
    if not param_lower:
        return False
    err_lower = str(exc).lower()
    if param_lower not in err_lower:
        return False
    return any(marker in err_lower for marker in (
        "unsupported parameter",
        "unsupported_parameter",
        "not supported",
        "does not support",
        "unknown parameter",
        "unrecognized request argument",
        "unrecognized parameter",
        "invalid parameter",
    ))


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    """Back-compat wrapper: detect API errors where the model rejects ``temperature``.

    Delegates to :func:`_is_unsupported_parameter_error`; kept as a separate
    public symbol because existing tests and call sites import it by name.
    """
    return _is_unsupported_parameter_error(exc, "temperature")


def _is_model_not_found_error(exc: Exception) -> bool:
    """Detect "the requested model doesn't exist" errors (404 / invalid model).

    This fires when a resolved model name is no longer served by the endpoint
    — most commonly when a long-lived process pinned a model that has since
    been dropped from the provider's catalog.  The endpoint returns 404 with
    a body like::

        Model 'gpt-5.4-mini' not found. The requested model does not exist
        in our configuration or catalog.

    Distinct from :func:`_is_payment_error` (which also matches some 404s for
    free-tier/credit language) — this one keys on "does not exist / not found /
    not a valid model" phrasing, and explicitly excludes the billing keywords
    that the payment path already owns so the two predicates don't overlap.
    """
    status = getattr(exc, "status_code", None)
    err_lower = str(exc).lower()
    # Billing/quota 404s belong to _is_payment_error — don't claim them here.
    if any(kw in err_lower for kw in (
        "credits", "insufficient funds", "billing", "out of funds",
        "balance_depleted", "no usable credits", "free tier", "free-tier",
        "not available on the free tier",
    )):
        return False
    if status not in {404, 400, None}:
        return False
    return any(kw in err_lower for kw in (
        "model does not exist",
        "does not exist in our configuration",
        "in our configuration or catalog",
        "is not a valid model",
        "no such model",
        "model not found",
        "the model `",            # OpenAI-style: "The model `X` does not exist"
        "model_not_found",
        "unknown model",
    ))


def _is_model_incompatible_error(exc: Exception) -> bool:
    """Detect "this route cannot serve this model" 400s (capability mismatch).

    Distinct from :func:`_is_model_not_found_error` (the model does not exist
    anywhere): here the model name is valid but the *current provider/account*
    is structurally unable to run it. The canonical case is a configured
    route that cannot run the main model — e.g. an ``openai-codex`` /
    ChatGPT-account route asked to compress a third-party-model
    conversation::

        Error code: 400 - {'detail': "The 'x' model is not supported
        when using Codex with a ChatGPT account."}

    The candidate authenticates fine and builds a client, so the auth and
    payment predicates don't fire and the call would otherwise raise and
    abort the whole auxiliary task (commonly compression — which then drops
    middle turns and churns the session, destroying the prompt cache).
    Treating it as a fallback-worthy capability error lets the chain skip the
    incapable route and continue to the next candidate, mirroring the
    context-window feasibility screen.

    Billing/quota 400s belong to :func:`_is_payment_error`; "model does not
    exist" 400s belong to :func:`_is_model_not_found_error`. This predicate
    explicitly excludes both so the three don't overlap.
    """
    status = getattr(exc, "status_code", None)
    if status not in {400, None}:
        return False
    err_lower = str(exc).lower()
    # Not-found 400s ("invalid model ID", "model does not exist") are owned by
    # _is_model_not_found_error. Billing/free-tier 400s are owned by the
    # payment path — key on the billing keywords directly here rather than
    # calling _is_payment_error(), because that predicate is status-gated
    # ({402,403,404,429,None}) and would not recognise a 400-coded billing
    # body, letting it leak into this capability bucket.
    if _is_model_not_found_error(exc):
        return False
    if any(kw in err_lower for kw in (
        "credits", "insufficient funds", "billing", "out of funds",
        "balance_depleted", "no usable credits", "payment required",
        "free tier", "free-tier", "not available on the free tier",
        "model_not_supported_on_free_tier", "quota",
    )):
        return False
    return any(kw in err_lower for kw in (
        "is not supported when using",   # codex/ChatGPT-account model gating
        "model is not supported",
        "not supported with this",
        "not supported for this account",
        "model_not_supported",
        "does not support this model",
        "unsupported model",
    ))


def _is_invalid_aux_response_error(exc: Exception) -> bool:
    """Detect provider responses that authenticated but cannot serve aux shape.

    Some OpenAI-compatible routes return HTTP 200 with an empty/malformed
    ChatCompletion instead of a normal provider error.  That is still a
    provider/model capability failure for auxiliary tasks: downstream callers
    need ``choices[0].message`` and should be able to continue through the
    same fallback path as explicit model-incompatibility errors.
    """
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return (
        "auxiliary " in msg
        and "llm returned invalid response" in msg
        and "choices[0].message" in msg
    )


def _evict_cached_clients(provider: str) -> None:
    """Drop cached auxiliary clients for a provider so fresh creds are used."""
    normalized = _normalize_aux_provider(provider)
    with _client_cache_lock:
        stale_keys = [
            key for key in _client_cache
            if _normalize_aux_provider(str(key[0])) == normalized
        ]
        for key in stale_keys:
            client = _client_cache.get(key, (None, None, None))[0]
            if client is not None:
                _close_cached_client(client)
            _client_cache.pop(key, None)


def _evict_cached_client_instance(target: Any) -> bool:
    """Drop the cache entry whose stored client is *target*.

    Used when a specific cached client has been poisoned (closed httpx
    transport after a timeout, broken streaming session, etc.) so the next
    auxiliary call rebuilds rather than reusing the dead instance.

    Walks both sync and async wrappers (``CodexAuxiliaryClient``,
    ``AsyncCodexAuxiliaryClient``, etc.) via
    their ``_real_client`` attribute so a timeout that closes the underlying
    ``OpenAI`` (or native provider) client evicts every cached shim that
    exposed it. Async wrappers must mirror their sync sibling's
    ``_real_client`` for this to work — otherwise the sync entry is evicted
    but the async entry survives and keeps reusing the dead transport.

    Returns True when at least one entry was evicted.
    """
    if target is None:
        return False
    evicted = False
    with _client_cache_lock:
        for key in list(_client_cache.keys()):
            entry = _client_cache.get(key)
            if entry is None:
                continue
            cached = entry[0]
            if cached is None:
                continue
            real = getattr(cached, "_real_client", None)
            if cached is target or real is target:
                del _client_cache[key]
                evicted = True
    return evicted


def _pool_cache_hint(
    provider: str,
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a stable cache discriminator for pooled providers."""
    normalized = _normalize_aux_provider(provider)
    if normalized == "auto":
        runtime = _normalize_main_runtime(main_runtime)
        normalized = _normalize_aux_provider(runtime.get("provider") or _read_main_provider())
    if normalized in {"", "auto", "custom"}:
        return ""
    entry = _peek_pool_entry(normalized)
    if entry is None:
        return ""
    entry_id = str(getattr(entry, "id", "") or "").strip()
    if not entry_id:
        return ""
    return f"{normalized}:{entry_id}"


def _pool_error_context(exc: Exception) -> Dict[str, Any]:
    status = getattr(exc, "status_code", None)
    payload: Dict[str, Any] = {"message": str(exc)}
    if status is not None:
        payload["status_code"] = status
    return payload


def _recoverable_pool_provider(
    resolved_provider: str,
    client: Any,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Infer which provider pool can recover the current auxiliary client."""
    normalized = _normalize_aux_provider(resolved_provider)
    if normalized not in {"", "auto", "custom"}:
        return normalized
    base = str(getattr(client, "base_url", "") or "")
    if base_url_host_matches(base, "chatgpt.com"):
        return "openai-codex"
    # For api_key providers not in the hardcoded list, match
    # the client base URL against all registered api_key providers so that
    # credential-pool rotation works for any provider the user configured.
    if main_runtime:
        rt = _normalize_main_runtime(main_runtime)
        rt_provider = rt.get("provider", "")
        if rt_provider and rt_provider not in {"", "auto", "custom"}:
            try:
                from pilotage_cli.auth import PROVIDER_REGISTRY
                pconfig = PROVIDER_REGISTRY.get(rt_provider)
                if pconfig and getattr(pconfig, "auth_type", None) == "api_key":
                    rt_base = str(getattr(pconfig, "inference_base_url", "") or "").rstrip("/")
                    if rt_base and base_url_host_matches(base, base_url_hostname(rt_base)):
                        return rt_provider
            except Exception:
                pass
    return None


def _recover_provider_pool(provider: str, exc: Exception, *, failed_api_key: str = "") -> bool:
    """Try same-provider credential-pool recovery for auxiliary calls.

    ``failed_api_key`` is the API key that was actually used for the failing
    request.  Passing it lets mark_exhausted_and_rotate identify the correct
    pool entry even when another process has already rotated the pool (which
    would leave current() as None, causing the wrong entry to be marked).
    """
    normalized = _normalize_aux_provider(provider)
    try:
        pool = load_pool(normalized)
    except Exception as load_exc:
        logger.debug("Auxiliary client: could not load pool for %s recovery: %s", normalized, load_exc)
        return False
    if not pool or not pool.has_credentials():
        return False

    status_code = getattr(exc, "status_code", None)
    error_context = _pool_error_context(exc)
    hint = failed_api_key or None

    if _is_auth_error(exc):
        refreshed = pool.try_refresh_current()
        if refreshed is not None:
            _evict_cached_clients(normalized)
            return True
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=status_code if status_code is not None else 401,
            error_context=error_context,
            api_key_hint=hint,
        )
        if next_entry is not None:
            _evict_cached_clients(normalized)
            return True
        return False

    if _is_payment_error(exc) or _is_rate_limit_error(exc):
        fallback_status = 402 if _is_payment_error(exc) else 429
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=status_code if status_code is not None else fallback_status,
            error_context=error_context,
            api_key_hint=hint,
        )
        if next_entry is not None:
            _evict_cached_clients(normalized)
            return True
    return False


def _retry_same_provider_sync(
    *,
    task: Optional[str],
    resolved_provider: str,
    resolved_model: Optional[str],
    resolved_base_url: Optional[str],
    resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str],
    main_runtime: Optional[Dict[str, Any]],
    final_model: Optional[str],
    messages: list,
    temperature: Optional[float],
    max_tokens: Optional[int],
    tools: Optional[list],
    effective_timeout: float,
    effective_extra_body: dict,
    reasoning_config: Optional[dict],
    extra_headers: Optional[Dict[str, str]] = None,
) -> Any:
    if task == "vision":
        effective_provider, retry_client, retry_model = resolve_vision_provider_client(
            provider=resolved_provider,
            model=final_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            async_mode=False,
        )
    else:
        retry_client, retry_model = _get_cached_client(
            resolved_provider,
            resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
            main_runtime=main_runtime,
        )
        effective_provider = _effective_provider_for_client(
            retry_client, resolved_provider,
        )
    if retry_client is None:
        raise RuntimeError(
            f"Auxiliary {task or 'call'}: provider {resolved_provider} could not be rebuilt after recovery"
        )

    retry_base = str(getattr(retry_client, "base_url", "") or "")
    retry_kwargs = _build_call_kwargs(
        effective_provider or resolved_provider,
        retry_model or final_model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        timeout=effective_timeout,
        extra_body=effective_extra_body,
        reasoning_config=reasoning_config,
        base_url=retry_base or resolved_base_url,
        task=task,
    )
    if extra_headers:
        retry_kwargs["extra_headers"] = dict(extra_headers)
    return _validate_llm_response(
        _relay_sync_completion(
            retry_client,
            retry_kwargs,
            provider=resolved_provider,
            api_mode=resolved_api_mode,
        ),
        task,
    )


async def _retry_same_provider_async(
    *,
    task: Optional[str],
    resolved_provider: str,
    resolved_model: Optional[str],
    resolved_base_url: Optional[str],
    resolved_api_key: Optional[str],
    resolved_api_mode: Optional[str],
    final_model: Optional[str],
    messages: list,
    temperature: Optional[float],
    max_tokens: Optional[int],
    tools: Optional[list],
    effective_timeout: float,
    effective_extra_body: dict,
    reasoning_config: Optional[dict],
    extra_headers: Optional[Dict[str, str]] = None,
) -> Any:
    if task == "vision":
        effective_provider, retry_client, retry_model = resolve_vision_provider_client(
            provider=resolved_provider,
            model=final_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            async_mode=True,
        )
    else:
        retry_client, retry_model = _get_cached_client(
            resolved_provider,
            resolved_model,
            async_mode=True,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
        )
        effective_provider = _effective_provider_for_client(
            retry_client, resolved_provider,
        )
    if retry_client is None:
        raise RuntimeError(
            f"Auxiliary {task or 'call'}: provider {resolved_provider} could not be rebuilt after recovery"
        )

    retry_base = str(getattr(retry_client, "base_url", "") or "")
    retry_kwargs = _build_call_kwargs(
        effective_provider or resolved_provider,
        retry_model or final_model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        timeout=effective_timeout,
        extra_body=effective_extra_body,
        reasoning_config=reasoning_config,
        base_url=retry_base or resolved_base_url,
        task=task,
    )
    # Preserve per-request attribution headers across the rebuilt-client
    # retry — see the sync variant above.
    if extra_headers:
        retry_kwargs["extra_headers"] = dict(extra_headers)
    return _validate_llm_response(
        await _relay_async_completion(
            retry_client,
            retry_kwargs,
            provider=resolved_provider,
            api_mode=resolved_api_mode,
        ),
        task,
    )


def _refresh_provider_credentials(provider: str) -> bool:
    """Refresh short-lived credentials for OAuth-backed auxiliary providers."""
    normalized = _normalize_aux_provider(provider)
    try:
        if normalized == "openai-codex":
            from pilotage_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=True)
            if not str(creds.get("api_key", "") or "").strip():
                return False
            _evict_cached_clients(normalized)
            return True
    except Exception as exc:
        logger.debug("Auxiliary provider credential refresh failed for %s: %s", normalized, exc)
        return False
    return False


def _auth_refresh_provider_for_route(
    resolved_provider: Optional[str],
    client_base_url: str,
) -> str:
    """Return the provider whose short-lived credentials should be refreshed.

    Auto-routed auxiliary calls keep ``resolved_provider == "auto"`` even
    after _get_cached_client() selects a concrete backend. Infer the backend
    from the selected client's base URL so auth refresh works for the
    auto → Codex route too.
    """
    normalized = _normalize_aux_provider(resolved_provider)
    if normalized and normalized != "auto":
        return normalized
    if base_url_host_matches(client_base_url, "chatgpt.com"):
        return "openai-codex"
    return normalized


def _resolve_auto_route(
    main_runtime: Optional[Dict[str, Any]] = None,
    task: Optional[str] = None,
) -> Tuple[Optional[OpenAI], Optional[str], str]:
    """Full auto-detection chain, including the selected provider identity.

    Priority:
      1. User's main provider + main model, regardless of provider type.
         This means auxiliary tasks (compression, vision, web extraction,
         session search, etc.) use the same model the user configured for
         chat.  Running aux tasks on the user's picked model keeps
         behavior predictable — no surprise
         switches to a cheap fallback model for side tasks.
    """
    global _stale_base_url_warned
    runtime = _normalize_main_runtime(main_runtime)
    runtime_provider = runtime.get("provider", "")
    runtime_model = str(runtime.get("model") or "")
    runtime_base_url = str(runtime.get("base_url") or "")
    runtime_api_key = runtime.get("api_key", "")
    runtime_api_mode = str(runtime.get("api_mode") or "")

    # ── Warn once if OPENAI_BASE_URL is set but config.yaml uses a named
    #    provider (not 'custom').  This catches the common "env poisoning"
    #    scenario where a user switches providers via `pilotage model` but the
    #    old OPENAI_BASE_URL lingers in ~/.pilotage/.env. ──
    if not _stale_base_url_warned:
        _env_base = os.getenv("OPENAI_BASE_URL", "").strip()
        _cfg_provider = runtime_provider or _read_main_provider()
        if (_env_base and _cfg_provider
                and _cfg_provider != "custom"
                and not _cfg_provider.startswith("custom:")):
            logger.warning(
                "OPENAI_BASE_URL is set (%s) but model.provider is '%s'. "
                "Auxiliary clients may route to the wrong endpoint. "
                "Run: pilotage model to reconfigure, or remove "
                "OPENAI_BASE_URL from ~/.pilotage/.env",
                _env_base, _cfg_provider,
            )
            _stale_base_url_warned = True

    # ── Step 1: main provider + main model → use them directly ──
    #
    # This is the primary aux backend for every user.  "auto" means
    # "use my main chat model for side tasks as well".  Explicit per-task
    # overrides set via
    # config.yaml (auxiliary.<task>.provider) still win over this.
    main_provider = str(runtime_provider or _read_main_provider() or "")
    main_model = str(runtime_model or _read_main_model() or "")

    # Latency-critical tasks can explicitly prefer the provider's registered
    # fast model over the main chat model. Titling is the only eligible task:
    # it names a visible sidebar row, produces ~8 tokens, and running it on a
    # frontier reasoning model costs seconds per new session. This remains an
    # opt-in because every settings surface defines "auto" as using the main
    # model; silently overriding that choice makes the selected model cosmetic.
    if _task_prefers_fast_model(task) and main_provider and main_provider not in {"auto", ""}:
        fast_model = _get_aux_model_for_provider(main_provider, prefer_fast=True)
        if fast_model and fast_model != main_model:
            logger.debug(
                "Auxiliary task %s: preferring fast model %s over main model %s",
                task, fast_model, main_model,
            )
            main_model = fast_model

    if (main_provider and main_model
            and main_provider not in {"auto", ""}):
        resolved_provider = main_provider
        explicit_base_url = runtime_base_url or None
        explicit_api_key = None
        if runtime_base_url and main_provider == "custom":
            # Anonymous custom endpoint (OPENAI_BASE_URL / config.model.base_url)
            # — pass through with explicit base_url + api_key.
            resolved_provider = "custom"
            explicit_base_url = runtime_base_url
            explicit_api_key = runtime_api_key or None
        elif main_provider.startswith("custom:"):
            # Named custom provider (custom_providers / providers dict entry).
            _has_named_entry = False
            try:
                from pilotage_cli.runtime_provider import _get_named_custom_provider
                _has_named_entry = _get_named_custom_provider(main_provider) is not None
            except ImportError:
                pass
            if _has_named_entry:
                # KEEP the full ``custom:<name>`` so resolve_provider_client
                # lands in the named-custom-provider arm — that arm honours the
                # entry's api_mode instead of rewriting the base URL.
                # Do NOT collapse to plain "custom"; that path rewrites the
                # base and routes through OpenAI chat.completions.
                # base_url and api_key come from the named entry itself, so
                # leave the explicit_* overrides unset.
                resolved_provider = main_provider
                explicit_base_url = None
            elif runtime_base_url:
                # Config-less named custom provider: the entry only
                # exists in the live runtime, so collapse to the anonymous
                # custom arm with the runtime endpoint + key.
                resolved_provider = "custom"
                explicit_base_url = runtime_base_url
                explicit_api_key = runtime_api_key or None
            elif runtime_api_key:
                explicit_api_key = runtime_api_key
        elif runtime_api_key:
            # Pin auxiliary to the same api_key as the active main chat session
            # so that a working key is reused instead of re-selecting from the pool
            # (which might pick a different, potentially exhausted key).
            explicit_api_key = runtime_api_key
        client, resolved = resolve_provider_client(
            resolved_provider,
            main_model,
            explicit_base_url=explicit_base_url,
            explicit_api_key=explicit_api_key,
            api_mode=runtime_api_mode or None,
        )
        if client is not None:
            logger.info("Auxiliary auto-detect: using main provider %s (%s)",
                        main_provider, resolved or main_model)
            return client, resolved or main_model, resolved_provider

    logger.warning(
        "Auxiliary auto-detect: no provider available. Compression, "
        "summarization, and memory flush will not work. Run: pilotage setup"
    )
    return None, None, ""


def _resolve_auto(
    main_runtime: Optional[Dict[str, Any]] = None,
    task: Optional[str] = None,
) -> Tuple[Optional[OpenAI], Optional[str]]:
    """Backward-compatible auto resolver for callers that only need client/model."""
    client, model, _provider = _resolve_auto_route(main_runtime=main_runtime, task=task)
    return client, model


def _tag_effective_provider(client: Any, provider: str) -> None:
    """Retain auto-routing identity on the client that survives cache reuse."""
    if client is None or not provider:
        return
    try:
        setattr(client, "_pilotage_aux_effective_provider", provider)
    except (AttributeError, TypeError):
        logger.debug(
            "Auxiliary client %s cannot retain effective provider %s",
            type(client).__name__, provider,
        )


def _effective_provider_for_client(client: Any, fallback: str) -> str:
    """Return the concrete provider selected for an auto-routed client."""
    effective_provider = getattr(client, "_pilotage_aux_effective_provider", "")
    if isinstance(effective_provider, str) and effective_provider:
        return effective_provider
    return str(fallback or "")


# ── Centralized Provider Router ─────────────────────────────────────────────
#
# resolve_provider_client() is the single entry point for creating a properly
# configured client given a (provider, model) pair.  It handles auth lookup,
# base URL resolution, provider-specific headers, and API format differences
# (Chat Completions vs Responses API for Codex).
#
# All auxiliary consumer code should go through this or the public helpers
# below — never look up auth env vars ad-hoc.


def _to_async_client(sync_client, model: str, is_vision: bool = False):
    """Convert a sync client to its async counterpart, preserving Codex routing.

    When ``is_vision=True`` and the underlying base URL is Copilot, the
    resulting async client carries the ``Copilot-Vision-Request: true``
    header so the request is routed to Copilot's vision-capable
    infrastructure (otherwise vision payloads silently time out).
    """
    from openai import AsyncOpenAI

    if isinstance(sync_client, _AuxProbeClientStub):
        return sync_client, model
    if isinstance(sync_client, CodexAuxiliaryClient):
        return AsyncCodexAuxiliaryClient(sync_client), model

    async_kwargs = {
        "api_key": sync_client.api_key,
        "base_url": str(sync_client.base_url),
    }
    sync_base_url = str(sync_client.base_url)
    # profile.default_headers for providers that declare client-level headers
    # on their ProviderProfile. Provider is inferred from the hostname.
    try:
        from agent.model_metadata import _infer_provider_from_url
        from providers import get_provider_profile as _gpf_async
        _inferred = _infer_provider_from_url(sync_base_url)
        if _inferred:
            _ph_async = _gpf_async(_inferred)
            if _ph_async and _ph_async.default_headers:
                async_kwargs["default_headers"] = dict(_ph_async.default_headers)
    except Exception:
        pass
    _merged_async = _apply_user_default_headers(async_kwargs.get("default_headers"))
    if _merged_async:
        async_kwargs["default_headers"] = _merged_async
    async_kwargs = {
        **_openai_http_client_kwargs(sync_base_url, async_mode=True),
        **async_kwargs,
    }
    # See _create_openai_client: disable SDK-internal retries so Pilotage owns
    # the auxiliary retry/timeout budget.
    async_kwargs.setdefault("max_retries", 0)
    return AsyncOpenAI(**async_kwargs), model


def _normalize_resolved_model(model_name: Optional[str], provider: str) -> Optional[str]:
    """Normalize a resolved model for the provider that will receive it."""
    if not model_name:
        return model_name
    try:
        from pilotage_cli.model_normalize import normalize_model_for_provider

        return normalize_model_for_provider(model_name, provider)
    except Exception:
        return model_name


def resolve_provider_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    raw_codex: bool = False,
    explicit_base_url: str = None,
    explicit_api_key: str = None,
    api_mode: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    task: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Central router: given a provider name and optional model, return a
    configured client with the correct auth, base URL, and API format.

    The returned client always exposes ``.chat.completions.create()`` — for
    Codex/Responses API providers, an adapter handles the translation
    transparently.

    Args:
        provider: Provider identifier.  One of:
            "openai", "openai-codex" (or "codex"),
            "custom" (OPENAI_BASE_URL + OPENAI_API_KEY),
            "auto" (full auto-detection chain).
        model: Model slug override.  If None, uses the provider's default
               auxiliary model.
        async_mode: If True, return an async-compatible client.
        raw_codex: If True, return a raw OpenAI client for Codex providers
            instead of wrapping in CodexAuxiliaryClient.  Use this when
            the caller needs direct access to responses.stream() (e.g.,
            the main agent loop).
        explicit_base_url: Optional direct OpenAI-compatible endpoint.
        explicit_api_key: Optional API key paired with explicit_base_url.
        api_mode: API mode override.  One of "chat_completions",
            "codex_responses", or None (auto-detect).  When set to
            "codex_responses", the client is wrapped in
            CodexAuxiliaryClient to route through the Responses API.

    Returns:
        (client, resolved_model) or (None, None) if auth is unavailable.
    """
    _validate_proxy_env_urls()
    # Preserve the original provider name before alias normalization so a
    # user-declared ``custom_providers`` entry whose name coincidentally
    # matches a built-in alias is still reachable via the named-custom
    # branch below.
    original_provider = (provider or "").strip().lower()
    # Normalise aliases
    provider = _normalize_aux_provider(provider)

    # Universal model-resolution fallback for concrete providers. ``auto`` is
    # intentionally excluded: `_resolve_auto(main_runtime=...)` returns the
    # model paired with the provider it actually selected. Pre-filling an auto
    # call from `_read_main_model()` can leak a stale process-global runtime
    # into a different provider (for example a foreign model slug on Codex OAuth)
    # and override that correctly resolved model.
    #
    # Concrete provider resolution order:
    #
    #   1. ``model`` argument (caller knew what they wanted)
    #   2. Provider's catalog default — cheap/fast model the provider
    #      registered via ``ProviderProfile.default_aux_model`` or the
    #      legacy ``_API_KEY_PROVIDER_AUX_MODELS_FALLBACK`` dict.  Empty
    #      string for OAuth-gated providers (openai-codex)
    #      whose accepted-model lists drift on the backend, so we don't
    #      pin a default that can silently rot.
    #   3. User's main model from ``model.model`` in config.yaml.  This is
    #      the load-bearing step for OAuth providers: the model the user
    #      configured is what gets used for title generation
    # instead of silently dropping to whatever Step-2 fallback.
    #      When the main provider is MoA, ``_read_main_model_for_aux()``
    #      substitutes the preset's aggregator model — the preset NAME is
    #      never a valid wire model id, so unset aux models default to the
    #      preset's acting model instead.
    #
    # Each provider branch below sees a non-empty ``model`` whenever the
    # user has *anything* configured — no provider-specific empty-model
    # guards needed.  When the user has NOTHING configured (fresh install,
    # main_model also empty), the branches still hit their own
    # missing-credentials returns and ``_resolve_auto`` falls through to
    # the Step-2 chain as before.
    #
    # Prefer explicit caller model, then provider-scoped aux model, then main model.
    # Do NOT pre-fill a blank ``auto`` request from the config/main default here.
    # ``auto`` has its own main-runtime resolver below; pre-filling first can pair
    # a stale configured model with a freshly resolved provider. Let
    # _resolve_auto()
    # return the actual current runtime model when the caller did not explicitly
    # request one. (# compression-current-model)
    if not model and provider != "auto":
        model = _get_aux_model_for_provider(provider) or _read_main_model_for_aux() or model

    def _needs_codex_wrap(client_obj, base_url_str: str, model_str: str) -> bool:
        """Decide if a plain OpenAI client should be wrapped for Responses API.

        Returns True when api_mode is explicitly "codex_responses", or when
        auto-detection (api.openai.com + codex-family model) suggests it.
        Already-wrapped clients (CodexAuxiliaryClient) are skipped.
        """
        if isinstance(client_obj, CodexAuxiliaryClient):
            return False
        if raw_codex:
            return False
        if provider == "actual":
            return True
        if api_mode == "codex_responses":
            return True
        # Auto-detect: api.openai.com + codex model name pattern
        if api_mode and api_mode != "codex_responses":
            return False  # explicit non-codex mode
        if base_url_hostname(base_url_str) == "api.openai.com":
            model_lower = (model_str or "").lower()
            if "codex" in model_lower:
                return True
        return False

    def _wrap_if_needed(client_obj, final_model_str: str, base_url_str: str = "",
                        api_key_str: str = ""):
        """Wrap a plain OpenAI client in the correct transport adapter.

        Handles one case:
        - ``CodexAuxiliaryClient`` when the endpoint needs the Responses API
          (explicit ``api_mode=codex_responses`` or api.openai.com + codex
          model name).
        Clients that are already specialized wrappers pass through unchanged.
        """
        if _needs_codex_wrap(client_obj, base_url_str, final_model_str):
            logger.debug(
                "resolve_provider_client: wrapping client in CodexAuxiliaryClient "
                "(api_mode=%s, model=%s, base_url=%s)",
                api_mode or "auto-detected", final_model_str,
                base_url_str[:60] if base_url_str else "")
            return CodexAuxiliaryClient(client_obj, final_model_str)
        return client_obj

    # ── Auto: try all providers in priority order ────────────────────
    if provider == "auto":
        client, resolved, effective_provider = _resolve_auto_route(
            main_runtime=main_runtime,
            task=task,
        )
        if client is None:
            return None, None
        # When auto-detection lands on a native provider, a vendor-prefixed
        # model override like "vendor/model" won't work.  Drop it and use
        # the provider's own default model instead.
        if model and "/" in model and resolved and "/" not in resolved:
            logger.debug(
                "Dropping vendor-prefixed model %r for native auxiliary "
                "provider (using %r instead)", model, resolved)
            model = None
        final_model = model or resolved
        routed_client, routed_model = (
            _to_async_client(client, final_model, is_vision=is_vision)
            if async_mode else (client, final_model)
        )
        _tag_effective_provider(routed_client, effective_provider)
        return routed_client, routed_model

    # ── OpenAI Codex (OAuth → Responses API) ─────────────────────────
    if provider == "openai-codex":
        if not model:
            logger.warning(
                "resolve_provider_client: openai-codex requested without a "
                "model; pass model explicitly (e.g. model.model in config.yaml "
                "or auxiliary.<task>.model for per-task aux routing)."
            )
            return None, None
        if raw_codex:
            # Return the raw OpenAI client for callers that need direct
            # access to responses.stream() (e.g., the main agent loop).
            codex_token = _read_codex_access_token()
            if not codex_token:
                logger.warning("resolve_provider_client: openai-codex requested "
                               "but no Codex OAuth token found (run: pilotage model)")
                return None, None
            final_model = _normalize_resolved_model(model, provider)
            raw_client = _create_openai_client(
                api_key=codex_token,
                base_url=_CODEX_AUX_BASE_URL,
                default_headers=_codex_cloudflare_headers(codex_token),
            )
            return (raw_client, final_model)
        # Standard path: wrap in CodexAuxiliaryClient adapter
        client, default = _build_codex_client(model)
        if client is None:
            logger.warning("resolve_provider_client: openai-codex requested "
                           "but no Codex OAuth token found (run: pilotage model)")
            return None, None
        final_model = _normalize_resolved_model(model or default, provider)
        return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                else (client, final_model))

    # ── Custom endpoint (OPENAI_BASE_URL + OPENAI_API_KEY) ───────────
    if provider == "custom":
        custom_base = ""
        custom_key = ""
        # Base passed to _wrap_if_needed for the transport decision.  It
        # normally equals custom_base.  Empty means "use custom_base".
        wrap_base = ""
        if explicit_base_url:
            custom_base = _to_openai_base_url(explicit_base_url).strip()
            custom_key = (
                (explicit_api_key or "").strip()
                or _scoped_key_env("OPENAI_API_KEY")
                or _read_main_api_key_if_same_host(custom_base)
                or "no-key-required"  # local servers don't need auth
            )
            if not custom_base:
                logger.warning(
                    "resolve_provider_client: explicit custom endpoint requested "
                    "but base_url is empty"
                )
                return None, None
        elif main_runtime:
            # When main_runtime carries a concrete base_url + api_key for a
            # named custom provider (custom:<name>), use it directly instead
            # of re-resolving from the bare "custom" provider name.
            # Re-resolution loses the provider name and falls back to
            # a wrong API-key provider — the main agent already
            # solved this, we just need to reuse its answer.
            _main_base = str(main_runtime.get("base_url") or "").strip().rstrip("/")
            _main_key = str(main_runtime.get("api_key") or "").strip()
            if _main_base and _main_key:
                custom_base = _main_base
                custom_key = _main_key
        if custom_base and custom_key:
            final_model = _normalize_resolved_model(
                model or (main_runtime.get("model") if main_runtime else None) or "gpt-4o-mini",
                provider,
            )
            extra = {}
            _clean_base, _dq = _extract_url_query_params(custom_base)
            if _dq:
                extra["default_query"] = _dq
            # profile.default_headers for providers that declare
            # client-level attribution headers on their profile.
            try:
                from providers import get_provider_profile as _gpf_custom
                _ph_custom = _gpf_custom(provider)
                if _ph_custom and _ph_custom.default_headers:
                    extra["default_headers"] = dict(_ph_custom.default_headers)
            except Exception:
                pass
            _merged_custom = _apply_user_default_headers(extra.get("default_headers"))
            if _merged_custom:
                extra["default_headers"] = _merged_custom
            client = _create_openai_client(api_key=custom_key, base_url=_clean_base, **extra)
            client = _wrap_if_needed(client, final_model, wrap_base or custom_base, custom_key)
            return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                    else (client, final_model))
        # Try custom first, then API-key providers (Codex excluded here:
        # falling through to Codex with no model is a stale-constant trap).
        for try_fn in (_try_custom_endpoint, _resolve_api_key_provider):
            client, default = try_fn()
            if client is not None:
                final_model = _normalize_resolved_model(model or default, provider)
                _cbase = str(getattr(client, "base_url", "") or "")
                # ``client.api_key`` may be a callable (Azure Foundry Entra
                # bearer provider). Pass empty string for the wrapper-detection
                # path — wrapping decisions are based on base_url + api_mode.
                _raw_ckey = getattr(client, "api_key", "")
                _ckey = "" if (callable(_raw_ckey) and not isinstance(_raw_ckey, str)) else str(_raw_ckey or "")
                client = _wrap_if_needed(client, final_model, _cbase, _ckey)
                return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                        else (client, final_model))
        logger.warning("resolve_provider_client: custom/main requested "
                       "but no endpoint credentials found")
        return None, None

    # ── Named custom providers (config.yaml providers dict / custom_providers list) ───
    try:
        from pilotage_cli.runtime_provider import _get_named_custom_provider
        # When the raw requested name is an alias of a built-in provider
        # and the user defined a ``custom_providers`` entry under that alias
        # name, the custom entry is the intended target — the built-in alias
        # rewriting would otherwise hijack the request.  Only preferred when
        # the raw name is an alias (not a canonical provider name) so custom
        # still defer to the built-in per `_get_named_custom_provider`'s guard.
        custom_entry = None
        if original_provider and original_provider != provider:
            custom_entry = _get_named_custom_provider(original_provider)
        if custom_entry is None:
            custom_entry = _get_named_custom_provider(provider)
        if custom_entry:
            custom_base = (custom_entry.get("base_url") or "").strip()
            custom_key = (custom_entry.get("api_key") or "").strip()
            custom_key_env = (custom_entry.get("key_env") or custom_entry.get("api_key_env") or "").strip()
            if not custom_key and custom_key_env:
                custom_key = _scoped_key_env(custom_key_env)
            custom_key = custom_key or "no-key-required"
            if custom_key == "no-key-required":
                logger.warning(
                    "resolve_provider_client: named custom provider %r has no resolvable "
                    "api_key — request will be sent with placeholder no-key-required "
                    "and will 401 on auth-required endpoints",
                    custom_entry.get("name") or provider,
                )
            # An explicit per-task api_mode override (from _resolve_task_provider_model)
            # wins; otherwise fall back to what the provider entry declared.
            entry_api_mode = (api_mode or custom_entry.get("api_mode") or "").strip()
            if custom_base:
                final_model = _normalize_resolved_model(
                    model
                    or custom_entry.get("model")
                    or (main_runtime.get("model") if main_runtime else None)
                    or _read_main_model_for_aux()
                    or "gpt-4o-mini",
                    provider,
                )
                openai_base = _to_openai_base_url(custom_base)
                raw_base_for_wrap = custom_base
                _clean_base2, _dq2 = _extract_url_query_params(openai_base)
                _extra2 = {"default_query": _dq2} if _dq2 else {}
                _headers2 = _apply_user_default_headers(_extra2.get("default_headers"))
                if _headers2:
                    _extra2["default_headers"] = _headers2
                logger.debug(
                    "resolve_provider_client: named custom provider %r (%s, api_mode=%s)",
                    provider, final_model, entry_api_mode or "chat_completions")
                client = _create_openai_client(api_key=custom_key, base_url=_clean_base2, **_extra2)
                # codex_responses or inherited auto-detect (via _wrap_if_needed).
                # _wrap_if_needed reads the closed-over `api_mode` (the task-level
                # override). Named-provider entry api_mode=codex_responses also
                # flows through here.
                if entry_api_mode == "codex_responses" and not isinstance(
                    client, CodexAuxiliaryClient
                ):
                    client = CodexAuxiliaryClient(client, final_model)
                else:
                    client = _wrap_if_needed(client, final_model, raw_base_for_wrap, custom_key)
                return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                        else (client, final_model))
            logger.warning(
                "resolve_provider_client: named custom provider %r has no base_url",
                provider)
            return None, None
    except ImportError:
        pass

    # ── Azure Foundry (delegates to runtime resolver for auth_mode-aware routing) ─
    #
    # The generic PROVIDER_REGISTRY path below uses
    # ``resolve_api_key_provider_credentials`` which only knows about the
    # static ``AZURE_FOUNDRY_API_KEY`` env var. That misses two important
    # cases for the ``azure-foundry`` provider:
    #
    #   1. ``model.auth_mode: entra_id`` — no static key exists; we need
    #      a callable bearer-token provider from ``azure_identity_adapter``.
    #   2. Non-default ``model.base_url`` (Foundry projects path) — the
    #      env-var-only resolver doesn't apply config-yaml-driven URL
    #      overrides.
    #
    # Delegate to the same runtime resolver the main agent uses so
    # auxiliary tasks (title generation, compression, vision, embedding,
    # session search) inherit the user's full Azure config.


    # ── API-key providers from PROVIDER_REGISTRY ─────────────────────
    try:
        from pilotage_cli.auth import (
            PROVIDER_REGISTRY,
            resolve_api_key_provider_credentials,
        )
    except ImportError:
        logger.debug("pilotage_cli.auth not available for provider %s", provider)
        return None, None

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig is None:
        # Demoted from logger.warning to debug; dedup keyed by provider name
        # so the first occurrence surfaces but repeated retries stay silent.
        if provider not in _LOGGED_UNKNOWN_PROVIDER_KEYS:
            _LOGGED_UNKNOWN_PROVIDER_KEYS.add(provider)
            logger.debug("resolve_provider_client: unknown provider %r", provider)
        return None, None

    if pconfig.auth_type == "api_key":
        creds = resolve_api_key_provider_credentials(provider)
        api_key = str(creds.get("api_key", "")).strip()
        # Honour an explicit api_key override (e.g. from a fallback_model entry
        # or a custom_providers entry) so callers that pass an explicit
        # credential can authenticate against endpoints where no built-in
        # credential is registered for this provider alias.
        if explicit_api_key:
            api_key = explicit_api_key.strip() or api_key
        raw_base_url = str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url
        if explicit_base_url:
            raw_base_url = explicit_base_url.strip().rstrip("/")
        if provider == "actual":
            try:
                from pilotage_cli.auth import (
                    ACTUAL_LOCAL_NOAUTH_PLACEHOLDER,
                    is_actual_local_base_url,
                    normalize_actual_base_url,
                )

                raw_base_url = normalize_actual_base_url(raw_base_url)
                if not api_key and is_actual_local_base_url(raw_base_url):
                    api_key = ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
            except Exception:
                pass
        if not api_key:
            tried_sources = list(pconfig.api_key_env_vars)
            logger.debug("resolve_provider_client: provider %s has no API "
                         "key configured (tried: %s)",
                         provider, ", ".join(tried_sources))
            return None, None

        base_url = _to_openai_base_url(raw_base_url)
        # Honour an explicit base_url override from the caller — used when a
        # fallback_model entry (or custom_providers lookup) routes through a
        # built-in provider name but targets a user-specified endpoint.
        if explicit_base_url:
            base_url = _to_openai_base_url(explicit_base_url.strip().rstrip("/"))

        default_model = _get_aux_model_for_provider(provider)
        final_model = _normalize_resolved_model(model or default_model, provider)

        # Provider-specific headers
        headers = {}
        # profile.default_headers for providers that declare client-level
        # attribution headers on their profile.
        try:
            from providers import get_provider_profile as _gpf_main
            _ph_main = _gpf_main(provider)
            if _ph_main and _ph_main.default_headers:
                headers.update(_ph_main.default_headers)
        except Exception:
            pass
        _merged_main = _apply_user_default_headers(headers)
        if _merged_main:
            headers = _merged_main
        client = _create_openai_client(api_key=api_key, base_url=base_url,
                        **({"default_headers": headers} if headers else {}))

        # Honor api_mode for any API-key provider (e.g. direct OpenAI with
        # codex-family models).
        client = _wrap_if_needed(client, final_model, raw_base_url, api_key)

        logger.debug("resolve_provider_client: %s (%s)", provider, final_model)
        return (_to_async_client(client, final_model, is_vision=is_vision) if async_mode
                else (client, final_model))

    if pconfig.auth_type == "external_process":
        if provider not in _LOGGED_UNSUPPORTED_EXTPROC_KEYS:
            _LOGGED_UNSUPPORTED_EXTPROC_KEYS.add(provider)
            logger.debug("resolve_provider_client: external-process provider %s not "
                         "directly supported", provider)
        return None, None

    elif pconfig.auth_type in {"oauth_device_code", "oauth_external"}:
        # OAuth providers — route through their specific try functions
        if provider == "openai-codex":
            return resolve_provider_client("openai-codex", model, async_mode)
        # Other OAuth providers not directly supported
        if provider not in _LOGGED_UNSUPPORTED_OAUTH_KEYS:
            _LOGGED_UNSUPPORTED_OAUTH_KEYS.add(provider)
            logger.debug("resolve_provider_client: OAuth provider %s not "
                         "directly supported, try 'auto'", provider)
        return None, None

    # Demoted from logger.warning to debug; dedup keyed on (auth_type,
    # provider) so the first occurrence surfaces (real schema-drift bug) but
    # per-call retries stay silent.
    _auth_dedup_key = (pconfig.auth_type, provider)
    if _auth_dedup_key not in _LOGGED_UNHANDLED_AUTHTYPE_KEYS:
        _LOGGED_UNHANDLED_AUTHTYPE_KEYS.add(_auth_dedup_key)
        logger.debug("resolve_provider_client: unhandled auth_type %s for %s",
                     pconfig.auth_type, provider)
    return None, None


# ── Public API ──────────────────────────────────────────────────────────────

def get_text_auxiliary_client(
    task: str = "",
    *,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, default_model_slug) for text-only auxiliary tasks.

    Args:
        task: Optional task name ("compression", "web_extract") to check
              for a task-specific provider override.

    Callers may override the returned model via config.yaml
    (e.g. auxiliary.compression.model, auxiliary.web_extract.model).
    """
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider,
        model=model,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
    )


def get_async_text_auxiliary_client(task: str = "", *, main_runtime: Optional[Dict[str, Any]] = None):
    """Return (async_client, model_slug) for async consumers.

    For standard providers returns (AsyncOpenAI, model). For Codex returns
    (AsyncCodexAuxiliaryClient, model) which wraps the Responses API.
    Returns (None, None) when no provider is available.
    """
    provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(task or None)
    return resolve_provider_client(
        provider,
        model=model,
        async_mode=True,
        explicit_base_url=base_url,
        explicit_api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
    )


_VISION_AUTO_PROVIDER_ORDER: tuple = ()


def _main_model_supports_vision(provider: str, model: Optional[str]) -> bool:
    """Return True when ``provider``/``model`` is known to accept image input.

    Used by the vision auto-detect chain to skip the user's main provider
    when it's known to be text-only (e.g. gpt-oss without vision).
    Without this guard, ``resolve_vision_provider_client(provider="auto")``
    would happily return the main-provider client and any subsequent image
    payload would surface as a cryptic provider-side error
    (``unknown variant `image_url`, expected `text```,).

    Returns True when capability lookup is unknown — preserves the historical
    behaviour of attempting the call, so providers we haven't catalogued yet
    don't silently regress to text-only.
    """
    try:
        from agent.image_routing import _lookup_supports_vision
        from pilotage_cli.config import load_config_readonly
    except ImportError:
        return True
    try:
        supports = _lookup_supports_vision(provider, model, load_config_readonly())
    except Exception:  # pragma: no cover - defensive
        return True
    if supports is None:
        # No capability data — keep current behaviour and let the call attempt
        # happen rather than silently skipping. This avoids false-positive
        # skips for new/custom providers.
        return True
    return bool(supports)


def _normalize_vision_provider(provider: Optional[str]) -> str:
    return _normalize_aux_provider(provider)


def _resolve_strict_vision_backend(
    provider: str,
    model: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    provider = _normalize_vision_provider(provider)
    if provider == "openai-codex":
        # Route through resolve_provider_client so the caller's explicit
        # model is used.  There is no safe default Codex model (shifting
        # allow-list); callers must specify via auxiliary.<task>.model.
        return resolve_provider_client("openai-codex", model, is_vision=True)
    if provider == "custom":
        return _try_custom_endpoint()
    return None, None


def _strict_vision_backend_available(provider: str) -> bool:
    return _resolve_strict_vision_backend(provider)[0] is not None


def get_available_vision_backends() -> List[str]:
    """Return the currently available vision backends in auto-selection order.

    Order: active provider → stop.  This is the source of truth for setup,
    tool gating, and runtime auto-routing of vision tasks.
    """
    available: List[str] = []
    # 1. Active provider — if the user configured a provider, try it first.
    main_provider = _read_main_provider()
    if main_provider and main_provider not in {"auto", ""}:
        if main_provider in _VISION_AUTO_PROVIDER_ORDER:
            if _strict_vision_backend_available(main_provider):
                available.append(main_provider)
        else:
            client, _ = resolve_provider_client(main_provider, _read_main_model())
            if client is not None:
                available.append(main_provider)
    return available


def resolve_vision_provider_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    async_mode: bool = False,
    main_runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """Resolve the client actually used for vision tasks.

    Direct endpoint overrides take precedence over provider selection. Explicit
    provider overrides still use the generic provider router for non-standard
    backends, so users can intentionally force experimental providers. Auto mode
    stays conservative and only tries vision backends known to work today.
    """
    runtime = _normalize_main_runtime(main_runtime)
    requested, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        "vision", provider, model, base_url, api_key
    )
    requested = _normalize_vision_provider(requested)

    def _finalize(resolved_provider: str, sync_client: Any, default_model: Optional[str]):
        if sync_client is None:
            return resolved_provider, None, None
        final_model = resolved_model or default_model
        if async_mode:
            async_client, async_model = _to_async_client(sync_client, final_model, is_vision=True)
            return resolved_provider, async_client, async_model
        return resolved_provider, sync_client, final_model

    if resolved_base_url:
        provider_for_base_override = (
            requested if requested and requested not in {"", "auto"} else "custom"
        )
        client, final_model = resolve_provider_client(
            provider_for_base_override,
            model=resolved_model,
            async_mode=async_mode,
            explicit_base_url=resolved_base_url,
            explicit_api_key=resolved_api_key,
            api_mode=resolved_api_mode,
            main_runtime=runtime,
        )
        if client is None:
            return provider_for_base_override, None, None
        return provider_for_base_override, client, final_model

    if requested == "auto":
        # Vision auto-detection order:
        #   1. User's main provider + main model (including aggregators).
        #      _PROVIDER_VISION_MODELS provides per-provider vision model
        #      overrides when the provider has a dedicated multimodal model
        #      that differs from the chat model, resolved live from the
        #      catalog (see :func:`_resolve_provider_vision_default`).
        #   2. Stop
        main_provider = str(runtime.get("provider") or _read_main_provider())
        main_model = str(runtime.get("model") or _read_main_model())
        if main_provider and main_provider not in {"auto", ""}:
            # A provider-specific vision default wins over the user's chat model:
            # static overrides and catalog-backed discovery both yield a
            # *known* vision-capable model, whereas the pinned chat model is
            # often NOT multimodal and _main_model_supports_vision can't be
            # trusted to catch that. Only fall back to the chat model when no
            # provider default is available (catalog unreachable).
            provider_vision_default = _resolve_provider_vision_default(main_provider)
            vision_model = provider_vision_default or main_model
            if main_provider in _PROVIDERS_WITHOUT_VISION:
                # This provider's endpoint does not accept image input at
                # all.  Skip it and fall through instead of returning a
                # client that will 404 on every vision request.
                logger.debug(
                    "Vision auto-detect: skipping main provider %s (no "
                    "vision support) — falling through to aggregator chain",
                    main_provider,
                )
            elif not _main_model_supports_vision(main_provider, vision_model):
                # The main model is known to be text-only (e.g. gpt-oss-120b
                # without vision). Building a client and sending
                # an image would produce a cryptic provider-side error like
                # ``unknown variant `image_url`, expected `text```.
                # Fall through to the aggregator chain instead.
                #
                # Only log the provider name (not the model) — mirrors the
                # sibling _PROVIDERS_WITHOUT_VISION branch above, and avoids
                # CodeQL py/clear-text-logging-sensitive-data heuristic false
                # positives on multi-value interpolations.
                logger.debug(
                    "Vision auto-detect: skipping main provider %s "
                    "(reports no vision capability) — falling through to "
                    "aggregator chain",
                    main_provider,
                )
            else:
                # Custom endpoints (``custom`` / ``custom:<name>``) carry no
                # built-in base_url/api_key — resolve_provider_client("custom")
                # would return None ("no endpoint credentials found") and the
                # whole chain would fall through to the aggregators, breaking
                # vision for every user on a custom provider that has no
                # separate ``auxiliary.vision`` block.  Recover the live main
                # endpoint that ``set_runtime_main()`` recorded for this turn so
                # Step 1 can build a working client.
                rpc_base_url = None
                rpc_api_key = None
                rpc_api_mode = resolved_api_mode
                if main_provider == "custom" or main_provider.startswith("custom:"):
                    runtime_base_url = runtime.get("base_url")
                    if runtime_base_url:
                        rpc_base_url = runtime_base_url
                        rpc_api_key = runtime.get("api_key") or None
                        rpc_api_mode = (
                            resolved_api_mode
                            or runtime.get("api_mode")
                            or None
                        )
                    else:
                        # No live runtime recorded (non-gateway caller): fall
                        # back to resolving the configured custom endpoint.
                        custom_base, custom_key, custom_mode = _resolve_custom_runtime()
                        if custom_base:
                            rpc_base_url = custom_base
                            rpc_api_key = custom_key
                            rpc_api_mode = resolved_api_mode or custom_mode or None
                rpc_client, rpc_model = resolve_provider_client(
                    main_provider, vision_model,
                    api_mode=rpc_api_mode,
                    explicit_base_url=rpc_base_url,
                    explicit_api_key=rpc_api_key,
                    main_runtime=runtime,
                    is_vision=True)
                if rpc_client is not None:
                    logger.info(
                        "Vision auto-detect: using main provider %s (%s)",
                        main_provider, rpc_model or vision_model,
                    )
                    return _finalize(
                        main_provider, rpc_client, rpc_model or vision_model)

        # Fall back through aggregators (uses their dedicated vision model,
        # not the user's main model) when main provider has no client.
        for candidate in _VISION_AUTO_PROVIDER_ORDER:
            if candidate == main_provider:
                continue  # already tried above
            sync_client, default_model = _resolve_strict_vision_backend(candidate)
            if sync_client is not None:
                return _finalize(candidate, sync_client, default_model)

        logger.debug("Auxiliary vision client: none available")
        return None, None, None

    if requested in _VISION_AUTO_PROVIDER_ORDER:
        sync_client, default_model = _resolve_strict_vision_backend(
            requested, resolved_model
        )
        return _finalize(requested, sync_client, default_model)

    client, final_model = _get_cached_client(requested, resolved_model, async_mode,
                                             api_mode=resolved_api_mode,
                                             main_runtime=runtime,
                                             is_vision=True)
    if client is None:
        return requested, None, None
    return requested, client, final_model


def get_auxiliary_extra_body() -> dict:
    """Return extra_body kwargs for auxiliary API calls."""
    return {}


def auxiliary_max_tokens_param(value: int, *, model: Optional[str] = None) -> dict:
    """Return the correct max tokens kwarg for the auxiliary client's provider.

    Most OpenAI-compatible endpoints use 'max_tokens'. Direct OpenAI with
    newer models (gpt-4.1, gpt-5+, o-series) requires 'max_completion_tokens'.
    The Codex adapter translates max_tokens internally, so we use max_tokens
    for it as well. Pass ``model`` so third-party OpenAI-compatible endpoints
    fronting the newer families are also recognised — URL-only detection
    misses the case where a custom base URL serves e.g. ``gpt-5.4``.
    """
    custom_base = _current_custom_base_url()
    # Use max_completion_tokens for direct OpenAI endpoints, which reject
    # max_tokens on newer o-series / GPT-5-style models.
    if (base_url_hostname(custom_base) or "") == "api.openai.com":
        return {"max_completion_tokens": value}
    # ...and for any caller serving a newer OpenAI-family model by name.
    if model_forces_max_completion_tokens(model):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}


# ── Centralized LLM Call API ────────────────────────────────────────────────
#
# call_llm() and async_call_llm() own the full request lifecycle:
#   1. Resolve provider + model from task config (or explicit args)
#   2. Get or create a cached client for that provider
#   3. Format request args for the provider + model (max_tokens handling, etc.)
#   4. Make the API call
#   5. Return the response
#
# Every auxiliary LLM consumer should use these instead of manually
# constructing clients and calling .chat.completions.create().

# Client cache: (provider, async_mode, base_url, api_key, api_mode, runtime_key) -> (client, default_model, loop)
# NOTE: loop identity is NOT part of the key.  On async cache hits we check
# whether the cached loop is the *current* loop; if not, the stale entry is
# replaced in-place.  This bounds cache growth to one entry per unique
# provider config rather than one per (config × event-loop), which previously
# caused unbounded fd accumulation in long-running gateway processes.
_client_cache: Dict[tuple, tuple] = {}
_client_cache_lock = threading.Lock()
_CLIENT_CACHE_MAX_SIZE = 64  # safety belt — evict oldest when exceeded


class _CallableCacheDiscriminator:
    """Hash a credential callback by identity without exposing its state."""

    __slots__ = ("_callback",)

    def __init__(self, callback: Any) -> None:
        # Retain the callback so its id cannot be reused while cached.
        self._callback = callback

    def __hash__(self) -> int:
        return id(self._callback)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _CallableCacheDiscriminator)
            and self._callback is other._callback
        )

    def __repr__(self) -> str:
        return "<callable-api-key>"


def _runtime_cache_discriminator(field: str, value: Any) -> Any:
    """Return a hashable, secret-safe runtime cache-key component."""
    if field == "api_key" and callable(value):
        return _CallableCacheDiscriminator(value)
    if field == "api_key" and isinstance(value, str) and value:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()
        return ("api-key-digest", digest)
    return value


def _client_cache_key(
    provider: str,
    *,
    async_mode: bool,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_mode: Optional[str] = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    task: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple:
    runtime = _normalize_main_runtime(main_runtime)
    runtime_key = tuple(
        _runtime_cache_discriminator(field, runtime.get(field, ""))
        for field in _MAIN_RUNTIME_FIELDS
    ) if provider == "auto" else ()
    # `auto` can now resolve through task-specific or main fallback policy,
    # so the task participates in the cache key. Non-auto providers keep the
    # old cache shape because the explicit provider/model tuple is sufficient.
    task_key = (
        (task or "", _task_prefers_fast_model(task))
        if provider == "auto"
        else ""
    )
    pool_hint = _pool_cache_hint(provider, main_runtime=main_runtime)
    # The model MUST participate in the key. Two concurrent auxiliary calls to
    # the SAME provider/base_url/key but DIFFERENT models (e.g. a MoA reference
    # fan-out running opus + gpt-5.5 in parallel threads) would otherwise share
    # one cache entry. On a cache MISS both build a client for the same key; the
    # second's _store_cached_client sees the first as the "old" entry and CLOSES
    # it — while the first call is still mid-request on it — yielding a spurious
    # APIConnectionError that fails the sibling advisor (root cause of the run2
    # double-advisor "Connection error" collapse). Keying on model gives each
    # model its own client, so concurrent fan-out calls never cross-close.
    model_key = model or runtime.get("model", "")
    api_key_key = _runtime_cache_discriminator("api_key", api_key or "")
    return (provider, async_mode, base_url or "", api_key_key, api_mode or "", runtime_key, is_vision, task_key, pool_hint, model_key)


def _store_cached_client(cache_key: tuple, client: Any, default_model: Optional[str], *, bound_loop: Any = None) -> None:
    if isinstance(client, _AuxProbeClientStub):
        # Probe stubs must never enter the cache — a runtime caller would
        # receive a non-functional client on the next cache hit.
        return
    with _client_cache_lock:
        old_entry = _client_cache.get(cache_key)
        if old_entry is not None and old_entry[0] is not client:
            _close_cached_client(old_entry[0])
        _client_cache[cache_key] = (client, default_model, bound_loop)


def neuter_async_httpx_del() -> None:
    """Monkey-patch ``AsyncHttpxClientWrapper.__del__`` to be a no-op.

    The OpenAI SDK's ``AsyncHttpxClientWrapper.__del__`` schedules
    ``self.aclose()`` via ``asyncio.get_running_loop().create_task()``.
    When an ``AsyncOpenAI`` client is garbage-collected while
    prompt_toolkit's event loop is running (the common CLI idle state),
    the ``aclose()`` task runs on prompt_toolkit's loop but the
    underlying TCP transport is bound to a *different* loop (the worker
    thread's loop that the client was originally created on).  If that
    loop is closed or its thread is dead, the transport's
    ``self._loop.call_soon()`` raises ``RuntimeError("Event loop is
    closed")``, which prompt_toolkit surfaces as "Unhandled exception
    in event loop ... Press ENTER to continue...".

    Neutering ``__del__`` is safe because:
    - Cached clients are explicitly cleaned via ``_force_close_async_httpx``
      on stale-loop detection and ``shutdown_cached_clients`` on exit.
    - Uncached clients' TCP connections are cleaned up by the OS when the
      process exits.
    - The OpenAI SDK itself marks this as a TODO (``# TODO(someday):
      support non asyncio runtimes here``).

    Call this once at CLI startup, before any ``AsyncOpenAI`` clients are
    created.
    """
    try:
        from openai._base_client import AsyncHttpxClientWrapper
        AsyncHttpxClientWrapper.__del__ = lambda self: None  # type: ignore[assignment]
    except (ImportError, AttributeError):
        pass  # Graceful degradation if the SDK changes its internals


def _force_close_async_httpx(client: Any) -> None:
    """Mark the httpx AsyncClient inside an AsyncOpenAI client as closed.

    This prevents ``AsyncHttpxClientWrapper.__del__`` from scheduling
    ``aclose()`` on a (potentially closed) event loop, which causes
    ``RuntimeError: Event loop is closed`` → prompt_toolkit's
    "Press ENTER to continue..." handler.

    We intentionally do NOT run the full async close path — the
    connections will be dropped by the OS when the process exits.
    """
    try:
        from httpx._client import ClientState
        inner = getattr(client, "_client", None)
        if inner is not None and not getattr(inner, "is_closed", True):
            inner._state = ClientState.CLOSED
    except Exception:
        pass


def _schedule_async_close(close_result: Any, client: Any) -> None:
    """Finish an async close without leaking an unawaited coroutine."""
    async def _await_close() -> None:
        try:
            await close_result
        except Exception:
            pass
        finally:
            _force_close_async_httpx(client)

    runner = _await_close()
    try:
        import asyncio as _aio

        try:
            loop = _aio.get_running_loop()
        except RuntimeError:
            _aio.run(runner)
        else:
            task = loop.create_task(runner)

            def _consume(completed_task) -> None:
                try:
                    completed_task.exception()
                except BaseException:
                    pass

            task.add_done_callback(_consume)
            runner = None
    except Exception:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass
        _force_close_async_httpx(client)


def _close_cached_client(client: Any, *, close_async: bool = False) -> None:
    """Close one cached client, awaiting async transports only when safe."""
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if not callable(close_fn):
        _force_close_async_httpx(client)
        return
    try:
        close_result = close_fn()
    except Exception:
        _force_close_async_httpx(client)
        return
    if inspect.isawaitable(close_result):
        if close_async:
            _schedule_async_close(close_result, client)
        else:
            # Do not await a client owned by another live event loop.
            # Closing the coroutine avoids an unawaited-coroutine warning;
            # the transport is still neutered for safe eventual GC.
            try:
                close_result.close()
            except Exception:
                pass
            _force_close_async_httpx(client)
        return
    _force_close_async_httpx(client)


def shutdown_cached_clients() -> None:
    """Close all cached clients (sync and async) to prevent event-loop errors.

    Call this during CLI shutdown, *before* the event loop is closed, to
    avoid ``AsyncHttpxClientWrapper.__del__`` raising on a dead loop.

    Snapshot and clear the cache under the lock, then close transports outside
    it. Async transport shutdown may block while an owner loop drains; holding
    the global cache lock during that wait stalls unrelated auxiliary callers
    and can turn teardown into a process-wide lock convoy.
    """
    with _client_cache_lock:
        clients = [
            (entry[0], entry[2])
            for entry in _client_cache.values()
            if entry[0] is not None
        ]
        _client_cache.clear()
    try:
        import asyncio as _aio

        running_loop = _aio.get_running_loop()
    except RuntimeError:
        running_loop = None
    for client, owner_loop in clients:
        # A live foreign loop owns its async transport. Calling its coroutine
        # on this thread can bind/close sockets from the wrong loop; neuter it
        # and let that owner finish teardown. Closed loops are safe to drain
        # locally, and the current loop can await its own client.
        close_async = owner_loop is not None and (
            owner_loop.is_closed() or owner_loop is running_loop
        )
        _close_cached_client(client, close_async=close_async)


def cleanup_stale_async_clients() -> None:
    """Force-close cached async clients whose event loop is closed.

    Call this after each agent turn to proactively clean up stale clients
    before GC can trigger ``AsyncHttpxClientWrapper.__del__`` on them.
    This is defense-in-depth — the primary fix is ``neuter_async_httpx_del``
    which disables ``__del__`` entirely.
    """
    stale_clients = []
    with _client_cache_lock:
        stale_keys = []
        for key, entry in _client_cache.items():
            client, _default, cached_loop = entry
            if cached_loop is not None and cached_loop.is_closed():
                stale_keys.append(key)
                stale_clients.append(client)
        for key in stale_keys:
            del _client_cache[key]
    for client in stale_clients:
        _close_cached_client(client, close_async=True)


def _cached_client_accepts_slash_models(client: Any, cached_default: Optional[str]) -> bool:
    """Best-effort check for cached clients that accept ``vendor/model`` IDs."""
    return bool(cached_default and "/" in cached_default)


def _compat_model(client: Any, model: Optional[str], cached_default: Optional[str]) -> Optional[str]:
    """Keep slash-bearing model IDs only for cached clients that support them.

    Mirrors the guard in resolve_provider_client() which is skipped on cache hits.
    """
    if model and "/" in model and not _cached_client_accepts_slash_models(client, cached_default):
        return cached_default
    return model or cached_default


def _get_cached_client(
    provider: str,
    model: str = None,
    async_mode: bool = False,
    base_url: str = None,
    api_key: str = None,
    api_mode: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    task: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Get or create a cached client for the given provider.

    Async clients (AsyncOpenAI) use httpx.AsyncClient internally, which
    binds to the event loop that was current when the client was created.
    Using such a client on a *different* loop causes deadlocks or
    RuntimeError.  To prevent cross-loop issues, the cache validates on
    every async hit that the cached loop is the *current, open* loop.
    If the loop changed (e.g. a new gateway worker-thread loop), the stale
    entry is replaced in-place rather than creating an additional entry.

    This keeps cache size bounded to one entry per unique provider config,
    preventing the fd-exhaustion that previously occurred in long-running
    gateways where recycled worker threads created unbounded entries.
    """
    # Resolve the current event loop for async clients so we can validate
    # cached entries.  Loop identity is NOT in the cache key — instead we
    # check at hit time whether the cached loop is still current and open.
    # This prevents unbounded cache growth from recycled worker-thread loops
    # while still guaranteeing we never reuse a client on the wrong loop
    # (which causes deadlocks, see).
    current_loop = None
    if async_mode:
        try:
            import asyncio as _aio
            current_loop = _aio.get_event_loop()
        except RuntimeError:
            pass
    runtime = _normalize_main_runtime(main_runtime)
    cache_key = _client_cache_key(
        provider,
        async_mode=async_mode,
        base_url=base_url,
        api_key=api_key,
        api_mode=api_mode,
        main_runtime=main_runtime,
        is_vision=is_vision,
        task=task,
        model=model,
    )
    with _client_cache_lock:
        if cache_key in _client_cache:
            cached_client, cached_default, cached_loop = _client_cache[cache_key]
            if async_mode:
                # Validate: the cached client must be bound to the CURRENT,
                # OPEN loop.  If the loop changed or was closed, the httpx
                # transport inside is dead — force-close and replace.
                loop_ok = (
                    cached_loop is not None
                    and cached_loop is current_loop
                    and not cached_loop.is_closed()
                )
                if loop_ok:
                    effective = _compat_model(cached_client, model, cached_default)
                    return cached_client, effective
                # Stale — evict and fall through to create a new client.
                # Only a client whose owner loop is closed may be awaited from
                # this thread; a live foreign loop remains force-neutered.
                owner_loop_closed = (
                    cached_loop is not None and cached_loop.is_closed()
                )
                _close_cached_client(cached_client, close_async=owner_loop_closed)
                del _client_cache[cache_key]
            else:
                effective = _compat_model(cached_client, model, cached_default)
                return cached_client, effective
    # Build outside the lock.
    # For pool-backed api_key providers, derive the active API key from the
    # pool entry rather than from env vars.  resolve_api_key_provider_credentials
    # always prefers env vars (first-entry bias), which bypasses pool rotation:
    # after key #1 is marked exhausted the retry would still get key #1 from
    # the env var and fail again, causing the retry2_err handler to mark key #2.
    effective_api_key = api_key
    if not effective_api_key:
        _pe = _peek_pool_entry(_normalize_aux_provider(provider))
        if _pe is not None:
            _pk = _pool_runtime_api_key(_pe)
            if _pk:
                effective_api_key = _pk
    client, default_model = resolve_provider_client(
        provider,
        model,
        async_mode,
        explicit_base_url=base_url,
        explicit_api_key=effective_api_key,
        api_mode=api_mode,
        main_runtime=runtime,
        is_vision=is_vision,
        task=task,
    )
    if client is not None:
        # For async clients, remember which loop they were created on so we
        # can detect stale entries later.
        bound_loop = current_loop
        with _client_cache_lock:
            if cache_key not in _client_cache:
                # Safety belt: if the cache has grown beyond the max, evict
                # the oldest entries (FIFO — dict preserves insertion order).
                # Do not close an evicted client here: another caller may be
                # mid-request with the object it obtained from this cache.
                # Dropping the cache reference lets normal refcount/GC cleanup
                # happen after in-flight users release it.
                while len(_client_cache) >= _CLIENT_CACHE_MAX_SIZE:
                    evict_key = next(iter(_client_cache))
                    del _client_cache[evict_key]
                _client_cache[cache_key] = (client, default_model, bound_loop)
            else:
                built_client = client
                client, default_model, _ = _client_cache[cache_key]
                # This concurrently built loser was never exposed to a caller,
                # so it is safe to close immediately.
                _close_cached_client(built_client, close_async=async_mode)
    return client, model or default_model


# Aliases that target direct REST APIs not modeled as first-class providers
# in PROVIDER_REGISTRY. Used for ``auxiliary.<task>.provider`` so users can
# write the obvious name and have it resolve to a working ``custom`` endpoint
# without needing to know our internal provider IDs.
#
# Why these specifically: PROVIDER_REGISTRY has ``openai-codex`` (OAuth) and
# ``custom`` (manual base_url + OPENAI_API_KEY) but no plain ``openai`` for
# direct API-key access. Users predictably type ``provider: openai`` and
# expect it to use OPENAI_API_KEY against api.openai.com. Previously this
# silently fell back to the user's main provider, sending OpenAI model names
# to a foreign endpoint and producing cryptic ``unknown variant 'image_url'``
# errors.
_AUX_DIRECT_API_BASE_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
}


def _resolve_task_provider_model(
    task: str = None,
    provider: str = None,
    model: str = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Determine provider + model for a call.

    Priority:
      1. Explicit provider/model/base_url/api_key args (always win)
      2. Config file (auxiliary.{task}.provider/model/base_url)
      3. "auto" (full auto-detection chain)

    Returns (provider, model, base_url, api_key, api_mode) where model may
    be None (use provider default). A bare base_url is treated as custom, but
    a first-class provider plus base_url keeps the provider identity so its
    auth, transport, and request-shaping behavior still apply. api_mode is one
    of "chat_completions", "codex_responses", or None (auto-detect).
    """
    cfg_provider = None
    cfg_model = None
    cfg_base_url = None
    cfg_api_key = None
    cfg_api_mode = None

    if task:
        task_config = _get_auxiliary_task_config(task)
        cfg_provider = str(task_config.get("provider", "")).strip() or None
        cfg_model = str(task_config.get("model", "")).strip() or None
        cfg_base_url = str(task_config.get("base_url", "")).strip() or None
        cfg_api_key = str(task_config.get("api_key", "")).strip() or None
        # Resolve key_env → env var when api_key is not set directly
        if not cfg_api_key:
            cfg_key_env = str(
                task_config.get("key_env") or task_config.get("api_key_env") or ""
            ).strip()
            if cfg_key_env:
                cfg_api_key = _scoped_key_env(cfg_key_env) or None
        cfg_api_mode = str(task_config.get("api_mode", "")).strip() or None

    # 'auto' is a sentinel meaning "inherit from main runtime / auto-detect", not
    # a literal model id. Without this, a config of `auxiliary.<task>.model: auto`
    # propagates the literal string "auto" to the wire, where the provider returns
    # a 200 OK with an error-text body (e.g. "the model 'auto' does not exist"),
    # which downstream consumers like ContextCompressor accept as the task output.
    # The provider-side 'auto' is handled in _resolve_auto() via main_runtime
    # fallback, so dropping cfg_model to None here lets that path do its job.
    #
    # The explicit `model` kwarg needs the identical normalization: MoA slots
    # (agent/moa_loop.py's _slot_runtime) forward a preset's `model:` field as
    # this explicit argument rather than through auxiliary.<task> config, so a
    # user-configured `model: auto` on a MoA reference/aggregator slot reaches
    # this function here, not as cfg_model. Only normalizing cfg_model let that
    # literal "auto" slip through via `model or cfg_model` below.
    if model and model.lower() == "auto":
        model = None
    if cfg_model and cfg_model.lower() == "auto":
        cfg_model = None

    resolved_model = model or cfg_model
    resolved_api_mode = cfg_api_mode

    # Convenience aliases for direct API-key endpoints that aren't first-class
    # providers (e.g. ``provider: openai`` → custom + api.openai.com/v1).
    # Applied to both explicit args and config-derived values. When the user
    # has already supplied a base_url we keep their endpoint but still rewrite
    # the provider to ``custom`` so resolution doesn't hit the
    # PROVIDER_REGISTRY-only path (which has no ``openai`` entry).
    def _expand_direct_api_alias(prov: Optional[str], existing_base: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not prov:
            return prov, existing_base
        target_base = _AUX_DIRECT_API_BASE_URLS.get(prov.strip().lower())
        if target_base is None:
            return prov, existing_base
        return "custom", existing_base or target_base

    def _preserve_provider_with_base_url(prov: Optional[str]) -> bool:
        normalized = str(prov or "").strip().lower()
        if normalized in {"", "auto", "custom"} or normalized.startswith("custom:"):
            return False
        try:
            from pilotage_cli.providers import get_provider

            return get_provider(normalized) is not None
        except Exception:
            # Keep the high-risk provider-backed routes safe even if provider
            # catalog loading is unavailable during early import/test paths.
            return normalized == "openai-codex"

    if provider:
        provider, base_url = _expand_direct_api_alias(provider, base_url)
    if cfg_provider:
        cfg_provider, cfg_base_url = _expand_direct_api_alias(cfg_provider, cfg_base_url)

    # An explicit provider arg without an explicit base_url must not bypass
    # the task's configured endpoint: adopt auxiliary.<task>.base_url/api_key
    # when the config targets the same provider (or names none), so the
    # early `if provider:` return below carries the configured endpoint
    # instead of falling through to main-runtime resolution.
    # An explicit "auto" is excluded — it means "inherit / auto-detect" and
    # must keep flowing through the existing auto-resolution chain.
    if provider and provider != "auto" and not base_url and cfg_base_url and cfg_provider in (None, provider):
        base_url = cfg_base_url
        if not api_key:
            api_key = cfg_api_key

    if base_url and _preserve_provider_with_base_url(provider):
        return provider, resolved_model, base_url, api_key, resolved_api_mode
    if base_url:
        return "custom", resolved_model, base_url, api_key, resolved_api_mode
    if provider:
        return provider, resolved_model, base_url, api_key, resolved_api_mode

    if task:
        # Config.yaml is the primary source for per-task overrides.
        if cfg_base_url and cfg_api_key:
            # Both base_url and api_key explicitly set → custom endpoint.
            return "custom", resolved_model, cfg_base_url, cfg_api_key, resolved_api_mode
        if cfg_base_url and cfg_provider and cfg_provider != "auto":
            # base_url set without api_key but with a known provider — use
            # the provider so it can resolve credentials from env vars
            # (e.g. OPENAI_API_KEY) instead of locking into "custom".
            return cfg_provider, resolved_model, cfg_base_url, None, resolved_api_mode
        if cfg_provider and cfg_provider != "auto":
            return cfg_provider, resolved_model, cfg_base_url, cfg_api_key, resolved_api_mode

        return "auto", resolved_model, None, None, resolved_api_mode

    return "auto", resolved_model, None, None, resolved_api_mode


_DEFAULT_AUX_TIMEOUT = 30.0

# Compression summarises large conversation histories; a reasoning auxiliary
# model (e.g. Codex / GPT-5.5) can legitimately take longer than the default
# ``auxiliary.compression.timeout`` (120 s), causing the stream to time out and
# the compressor to fall back to the deterministic context marker.
# This is a bounded *floor* applied only to config-derived compression timeouts
# — it does not affect other auxiliary tasks and does not override an explicit
# per-call ``timeout=``.  A floor is harmless for fast compression models
# (they finish before the deadline) and is a minimum, so a higher config value
# is kept unchanged.
_COMPRESSION_TIMEOUT_FLOOR_SECONDS = 300.0


def _get_auxiliary_task_config(task: str) -> Dict[str, Any]:
    """Return the config dict for auxiliary.<task>, or {} when unavailable.

    For plugin-registered auxiliary tasks (see
    :meth:`pilotage_cli.plugins.PluginContext.register_auxiliary_task`) the
    plugin's declared *defaults* are layered underneath the user's config
    so an unconfigured plugin task still works:

        plugin defaults  ←  config.yaml auxiliary.<task>  (user wins)

    Built-in tasks ignore this path (their defaults live in DEFAULT_CONFIG).
    """
    if not task:
        return {}
    try:
        from pilotage_cli.config import load_config_readonly
        config = load_config_readonly()
    except ImportError:
        return {}
    aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    task_config = aux.get(task, {}) if isinstance(aux, dict) else {}
    if not isinstance(task_config, dict):
        task_config = {}

    # Layer plugin-declared defaults underneath user config so
    # ctx.register_auxiliary_task(defaults={...}) takes effect without
    # forcing the user to write config.yaml entries.
    try:
        from pilotage_cli.plugins import get_plugin_auxiliary_tasks
        for _entry in get_plugin_auxiliary_tasks():
            if _entry.get("key") == task:
                _defaults = _entry.get("defaults") or {}
                if isinstance(_defaults, dict):
                    merged = dict(_defaults)
                    merged.update(task_config)
                    return merged
                break
    except Exception:
        # Plugin discovery failure must not break aux task config reads.
        pass

    return task_config


def _get_task_timeout(task: str, default: float = _DEFAULT_AUX_TIMEOUT) -> float:
    """Read timeout from auxiliary.{task}.timeout in config, falling back to *default*."""
    if not task:
        return default
    task_config = _get_auxiliary_task_config(task)
    raw = task_config.get("timeout")
    if raw is not None:
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass
    return default


def _effective_aux_timeout(task: str, timeout: Optional[float]) -> float:
    """Resolve the effective timeout for an auxiliary LLM call.

    Uses the caller-provided ``timeout`` when given; otherwise reads
    ``auxiliary.{task}.timeout`` from config via :func:`_get_task_timeout`.
    For the ``compression`` task only, applies a bounded floor so a reasoning
    model summarising a large context is not cut off by the default timeout
. The floor is intentionally skipped when the caller passes an
    explicit ``timeout=`` — explicit per-call deadlines are always honoured —
    and it is a minimum (``max``), so a config value already above it is kept.
    """
    effective = timeout if timeout is not None else _get_task_timeout(task)
    if timeout is None and task == "compression":
        effective = max(effective, _COMPRESSION_TIMEOUT_FLOOR_SECONDS)
    return effective


def _get_task_extra_body(task: str) -> Dict[str, Any]:
    """Read auxiliary.<task>.extra_body and return a shallow copy when valid.

    Also folds in ``auxiliary.<task>.reasoning_effort`` as an
    ``extra_body.reasoning`` config dict ({"enabled": ..., "effort": ...})
    when set. An explicit ``extra_body.reasoning`` in config wins over the
    ``reasoning_effort`` shorthand (it is the more specific wire control).
    Downstream, each wire already translates ``extra_body.reasoning``:
    chat.completions passes it through, the Codex Responses adapter maps it
    to top-level ``reasoning``/``include``.

    MoA tasks are excluded by design: reasoning depth for MoA is a per-slot
    setting in the MoA preset (``moa.presets.<name>.reference_models[].
    reasoning_effort`` / ``aggregator.reasoning_effort``), not an
    auxiliary-task knob — an ensemble-wide value would override the
    per-slot ones.
    """
    task_config = _get_auxiliary_task_config(task)
    raw = task_config.get("extra_body")
    result = dict(raw) if isinstance(raw, dict) else {}
    if "reasoning" not in result:
        effort = task_config.get("reasoning_effort")
        if effort is not None and effort != "":
            from pilotage_constants import parse_reasoning_effort
            parsed = parse_reasoning_effort(effort)
            if parsed is not None:
                result["reasoning"] = parsed
            else:
                logger.warning(
                    "auxiliary.%s.reasoning_effort %r is not a valid level "
                    "(none, minimal, low, medium, high, xhigh, max, ultra) — ignoring",
                    task, effort,
                )
    return result


# ---------------------------------------------------------------------------
# Per-task concurrency limiting
# ---------------------------------------------------------------------------
# Background auxiliary work (title generation, context compression, etc.) can
# spawn unbounded concurrent LLM calls when many sessions are active. During
# provider incidents each call also retries / fans out across the fallback
# chain, multiplying request volume on already-degraded endpoints. A per-task
# semaphore caps in-flight calls so retry amplification stays bounded.

_aux_sync_semaphores: Dict[str, Tuple[int, threading.BoundedSemaphore]] = {}
_aux_async_semaphores: Dict[Tuple[str, int], Tuple[int, Any]] = {}
_aux_sem_lock = threading.Lock()


def _get_task_max_concurrency(task: Optional[str]) -> Optional[int]:
    """Return ``auxiliary.<task>.max_concurrency`` as a positive int, or None."""
    if not task or task == "vision":
        # Vision already uses this key for its encode/resize CPU worker pool;
        # its LLM calls deliberately remain concurrent.
        return None
    raw = _get_auxiliary_task_config(task).get("max_concurrency")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _acquire_sync_aux_semaphore(task: Optional[str]) -> Optional[threading.BoundedSemaphore]:
    """Get a per-task sync semaphore, rebuilding it after a config change."""
    limit = _get_task_max_concurrency(task)
    if limit is None:
        return None
    with _aux_sem_lock:
        entry = _aux_sync_semaphores.get(task)
        if entry is None or entry[0] != limit:
            semaphore = threading.BoundedSemaphore(limit)
            _aux_sync_semaphores[task] = (limit, semaphore)
            return semaphore
        return entry[1]


def _acquire_async_aux_semaphore(task: Optional[str]):
    """Get a per-task, per-event-loop async semaphore after config lookup."""
    limit = _get_task_max_concurrency(task)
    if limit is None:
        return None
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    key = (task, id(loop))
    with _aux_sem_lock:
        entry = _aux_async_semaphores.get(key)
        if entry is None or entry[0] != limit:
            semaphore = asyncio.Semaphore(limit)
            _aux_async_semaphores[key] = (limit, semaphore)
            return semaphore
        return entry[1]


def _reset_aux_semaphores() -> None:
    """Drop cached semaphores (test helper)."""
    with _aux_sem_lock:
        _aux_sync_semaphores.clear()
        _aux_async_semaphores.clear()


_PROFILE_REASONING_KEYS = {
    "reasoning",
    "reasoning_effort",
    "thinking",
    "thinking_config",
    "thinkingconfig",
    "thinking_budget",
    "thinkingbudget",
    "enable_thinking",
    "think",
    "verbosity",
}


def _contains_profile_reasoning_fields(value: Any) -> bool:
    """Return whether a profile payload contains a reasoning wire control."""
    if not isinstance(value, dict):
        return False
    for key, nested in value.items():
        normalized = str(key).strip().lower()
        if normalized in _PROFILE_REASONING_KEYS:
            return True
        if _contains_profile_reasoning_fields(nested):
            return True
    return False


def _build_call_kwargs(
    provider: str,
    model: str,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    timeout: float = 30.0,
    extra_body: Optional[dict] = None,
    reasoning_config: Optional[dict] = None,
    base_url: Optional[str] = None,
    task: Optional[str] = None,
) -> dict:
    """Build kwargs for .chat.completions.create() with model/provider adjustments."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
    }

    # Opus 4.7+ rejects any non-default temperature/top_p/top_k — silently
    # drop here so auxiliary callers that hardcode temperature (e.g. 0 on
    # structured-JSON extraction) don't 400 the moment
    # the aux model is flipped to 4.7.
    if temperature is not None:
        kwargs["temperature"] = temperature

    # max_tokens is deliberately not forwarded: chat-completions providers
    # treat an omitted cap as "use the model's max output", which is what
    # auxiliary tasks (compression, titles, vision) want.

    if tools:
        # Defensive dedup: providers like Google Vertex, Azure, and Bedrock
        # reject requests with duplicate tool names (HTTP 400).  The upstream
        # injection paths (run_agent.py) already dedup, but this guard
        # converts a hard API failure into a warning if an upstream regression
        # reintroduces duplicates. See:
        _seen: set = set()
        _deduped: list = []
        for _t in tools:
            _tname = (_t.get("function") or {}).get("name", "")
            if _tname and _tname in _seen:
                logger.warning(
                    "_build_call_kwargs: duplicate tool name '%s' removed "
                    "(provider=%s model=%s)",
                    _tname, provider, model,
                )
                continue
            if _tname:
                _seen.add(_tname)
            _deduped.append(_t)
        kwargs["tools"] = _deduped

    # Build provider-aware reasoning kwargs through the same profile hooks used
    # by the standard chat-completions transport. Some providers require
    # top-level controls (``reasoning_effort``), others use nested body
    # fields (``thinking_config``) or ``extra_body.reasoning``.
    # Profiles are the source of truth for those wire
    # shapes. Providers without a reasoning-aware profile retain the generic
    # ``extra_body.reasoning`` fallback used by Codex-compatible adapters.
    effective_base = base_url or (
        _current_custom_base_url() if provider == "custom" else ""
    )
    profile_body: Dict[str, Any] = {}
    profile_reasoning_extra: Dict[str, Any] = {}
    profile_top_level: Dict[str, Any] = {}
    profile_handles_reasoning = False
    try:
        from providers import get_provider_profile
        from providers.base import ProviderProfile

        profile = get_provider_profile(str(provider or "").strip().lower())
        if profile is not None:
            profile_body = profile.build_extra_body(
                model=model,
                base_url=effective_base,
                reasoning_config=reasoning_config,
            ) or {}
            profile_reasoning_extra, profile_top_level = (
                profile.build_api_kwargs_extras(
                    reasoning_config=reasoning_config,
                    supports_reasoning=reasoning_config is not None,
                    model=model,
                    base_url=effective_base,
                )
            )
            profile_reasoning_extra = profile_reasoning_extra or {}
            profile_top_level = profile_top_level or {}
            profile_handles_reasoning = (
                type(profile).build_api_kwargs_extras
                is not ProviderProfile.build_api_kwargs_extras
                or _contains_profile_reasoning_fields(profile_body)
                or _contains_profile_reasoning_fields(profile_reasoning_extra)
                or _contains_profile_reasoning_fields(profile_top_level)
            )
    except Exception as exc:
        logger.debug(
            "_build_call_kwargs: provider profile projection failed for %s: %s",
            provider,
            exc,
        )

    kwargs.update(profile_top_level)
    merged_extra = dict(extra_body or {})
    merged_extra.update(profile_body)
    merged_extra.update(profile_reasoning_extra)
    if (
        reasoning_config
        and isinstance(reasoning_config, dict)
        and not profile_handles_reasoning
    ):
        if reasoning_config.get("enabled") is False:
            merged_extra["reasoning"] = {"enabled": False}
        else:
            effort = reasoning_config.get("effort") or "medium"
            merged_extra["reasoning"] = {"enabled": True, "effort": effort}
    if merged_extra:
        kwargs["extra_body"] = merged_extra

    return kwargs


def _validate_llm_response(
    response: Any,
    task: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    """Validate that an LLM response has the expected .choices[0].message shape.

    Fails fast with a clear error instead of letting malformed payloads
    propagate to downstream consumers where they crash with misleading
    AttributeError (e.g. "'str' object has no attribute 'choices'").

    See.

    Also the single accounting chokepoint for auxiliary usage: every
    successful non-streaming aux response passes through here exactly once,
    so token usage is recorded against the ambient session context published
    by the agent loop (``agent.aux_accounting``,). Recording is
    best-effort and never affects validation. *provider*/*base_url* are
    optional accounting hints — fallback-path calls omit them and the row
    keeps the model (read from the response itself) with an empty route.
    """
    if response is None:
        raise RuntimeError(
            f"Auxiliary {task or 'call'}: LLM returned None response"
        )
    from agent.aux_accounting import record_aux_usage
    record_aux_usage(response, task, provider=provider, base_url=base_url)
    # Allow SimpleNamespace responses from adapters (CodexAuxiliaryClient)
    # — they have .choices[0].message.
    try:
        choices = response.choices
        if not choices or not hasattr(choices[0], "message"):
            raise AttributeError("missing choices[0].message")
    except (AttributeError, TypeError, IndexError) as exc:
        recovered = _recover_aux_response_message(response)
        if recovered is not None:
            _record_relay_auxiliary_response_model(response)
            _complete_relay_auxiliary_call()
            return recovered
        response_type = type(response).__name__
        response_preview = str(response)[:120]
        raise RuntimeError(
            f"Auxiliary {task or 'call'}: LLM returned invalid response "
            f"(type={response_type}): {response_preview!r}. "
            f"Expected object with .choices[0].message — check provider "
            f"adapter or custom endpoint compatibility."
        ) from exc
    _record_relay_auxiliary_response_model(response)
    _complete_relay_auxiliary_call()
    return response


def _complete_relay_auxiliary_call(*, outcome: str = "success") -> None:
    """Close one auxiliary logical call after acceptance or terminal failure."""
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return
    from agent import relay_llm

    relay_llm.complete_logical_call(
        str(context.get("request_id") or ""),
        outcome=outcome,
        model_name=str(context.get("model") or "unknown"),
        provider_name=str(context.get("provider") or "auxiliary"),
        response_model_name=context.get("response_model"),
    )


def _record_relay_auxiliary_response_model(response: Any) -> None:
    """Retain the provider-reported model for terminal route attribution."""
    context = _RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return
    if isinstance(response, dict):
        model = response.get("model")
    else:
        model = getattr(response, "model", None)
    if isinstance(model, str) and model.strip():
        context["response_model"] = model


def _fail_relay_auxiliary_call() -> None:
    """Close a terminally failed call without replacing its original error."""
    try:
        _complete_relay_auxiliary_call(outcome="failed")
    except Exception:
        logger.warning(
            "Relay auxiliary failure finalization failed",
            exc_info=True,
        )


def _recover_aux_response_message(response: Any) -> Optional[Any]:
    """Synthesize chat-completions shape from Responses-style text fields.

    Auxiliary callers consume ``choices[0].message``.  Some compatible
    endpoints return text outside ``choices`` (for example ``output_text`` or
    ``output`` items).  Preserve that response before declaring it malformed.
    """
    text = _extract_aux_response_text(response)
    if not text:
        return None

    choice = SimpleNamespace(
        message=SimpleNamespace(content=text),
        finish_reason=getattr(response, "finish_reason", None) or "stop",
    )
    try:
        response.choices = [choice]
        return response
    except Exception:
        return SimpleNamespace(
            id=getattr(response, "id", ""),
            model=getattr(response, "model", ""),
            object=getattr(response, "object", "chat.completion"),
            choices=[choice],
            usage=getattr(response, "usage", None),
        )


def _extract_aux_response_text(response: Any) -> str:
    output_text = _obj_get(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = _obj_get(response, "output")
    if not isinstance(output, list):
        return ""

    parts: List[str] = []
    for item in output:
        item_type = _obj_get(item, "type")
        if item_type and item_type != "message":
            continue
        for part in (_obj_get(item, "content") or []):
            part_type = _obj_get(part, "type")
            if part_type in {"output_text", "text", None}:
                text = _obj_get(part, "text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    return "\n".join(parts).strip()


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    value = getattr(obj, key, default)
    if value is default and isinstance(obj, dict):
        value = obj.get(key, default)
    return value


# ── Streamed aggregation for progress-hooked auxiliary calls ─────────────
# When a forward-progress hook is installed (aux_progress_hook — today only
# by context compression), the primary chat.completions attempt is upgraded
# to a streamed request that is aggregated back into a complete response.
# Two effects, both deliberate:
#   1. The configured ``timeout`` becomes an INTER-CHUNK idle timeout instead
#      of a total budget (httpx applies the read timeout per stream read), so
#      a slow-but-generating summary model is never killed mid-generation
#      while tokens are moving — only a genuinely silent connection dies.
#   2. Every arriving chunk ticks the progress hook, letting outer watchdogs
#      (gateway session hygiene) extend their deadlines on liveness instead
#      of guessing with a fixed wall clock.
# A total ceiling still bounds the pathological 1-token-per-idle-window
# stream; see _aux_stream_total_ceiling().

_AUX_STREAM_CEILING_FLOOR_SECONDS = 600.0
_AUX_STREAM_CEILING_MULTIPLIER = 4.0


def _aux_stream_total_ceiling(effective_timeout: Optional[float]) -> float:
    """Absolute wall-clock bound for a progress-hooked streamed aux call.

    Generous by design — the idle timeout is the real guard; this only stops
    a degenerate stream that trickles one token per idle window forever.
    """
    try:
        timeout = float(effective_timeout) if effective_timeout is not None else 0.0
    except (TypeError, ValueError):
        timeout = 0.0
    return max(_AUX_STREAM_CEILING_FLOOR_SECONDS,
               _AUX_STREAM_CEILING_MULTIPLIER * timeout)


def _client_streams_internally(client: Any) -> bool:
    """Wire adapters that consume a stream inside .create() already tick the
    progress hook themselves (Codex per SSE event) and do not accept
    chat-completions ``stream=True`` semantics from us."""
    return isinstance(client, CodexAuxiliaryClient)


def _is_streaming_rejected_error(exc: Exception) -> bool:
    """Provider explicitly refused a streamed chat.completions request."""
    err = str(exc).lower()
    if "stream_options" in err:
        return True
    return "stream" in err and (
        "not supported" in err
        or "unsupported" in err
        or "not allowed" in err
        or "disabled" in err
    )


def _provider_requires_stream(provider: str, base_url: Optional[str]) -> bool:
    """Detect providers that only accept streaming (non-stream = HTTP 400).

    Some OpenAI-compatible endpoints reject non-streaming chat requests
    outright. The main conversation loop already streams, so interactive
    chat works; auxiliary tasks (title generation, compression, web extract)
    used the non-streaming path and failed on every call. When this returns
    True the auxiliary client sends ``stream=True`` and aggregates the chunks
    itself (see :func:`_aggregate_chat_stream`).

    Beyond the known-host list, users can mark ANY custom endpoint as
    stream-only via ``auxiliary.stream_only_base_urls`` in config.yaml
    (list of substrings matched against the endpoint URL).
    """
    _url = str(base_url or "").lower()
    if not _url:
        return False
    try:
        from pilotage_cli.config import load_config
        aux_cfg = (load_config() or {}).get("auxiliary", {})
        markers = aux_cfg.get("stream_only_base_urls") or []
        if isinstance(markers, (list, tuple)):
            for marker in markers:
                if isinstance(marker, str) and marker.strip() and marker.strip().lower() in _url:
                    return True
    except Exception:
        # Config read is best-effort; never break an aux call over it.
        pass
    return False


def _create_with_progress(
    client: Any,
    kwargs: Dict[str, Any],
    task: Optional[str] = None,
    *,
    force_stream: bool = False,
) -> Any:
    """chat.completions.create() that streams when a progress hook is active
    or the provider only accepts streamed requests.

    Behavior is byte-for-byte identical to a plain ``create(**kwargs)`` when
    neither trigger applies (every existing caller/task) or when the client's
    wire adapter streams internally. With a hook + a chunk-capable client,
    the request is sent with ``stream=True`` and aggregated, ticking the hook
    per chunk — so the configured ``timeout`` acts per stream read (idle)
    rather than as a total budget, and outer liveness watchdogs see tokens
    moving. Providers that reject the streamed request fall back to
    the plain non-streaming call — except under ``force_stream``, where a
    stream-only provider rejects the plain call by definition, so the
    original error is surfaced to the normal recovery chains instead.
    """
    _notify_aux_progress()  # request dispatched counts as progress
    if (not _aux_progress_active() and not force_stream) or _client_streams_internally(client):
        return client.chat.completions.create(**kwargs)

    total_ceiling = _aux_stream_total_ceiling(kwargs.get("timeout"))
    stream_kwargs = dict(kwargs)
    stream_kwargs["stream"] = True
    stream_kwargs["stream_options"] = {"include_usage": True}
    try:
        chunks = client.chat.completions.create(**stream_kwargs)
    except Exception as exc:
        # Genuine provider failures (auth, credit, rate limit, network) are
        # not streaming's fault — surface them unchanged so the existing
        # recovery chains (credential refresh, pool rotation, provider
        # fallback) see the same error they would on a plain call.
        if (
            force_stream
            or _is_transient_transport_error(exc)
            or _is_auth_error(exc)
            or _is_payment_error(exc)
            or _is_rate_limit_error(exc)
        ):
            raise
        # Anything else may be a streaming-specific rejection (explicit
        # "stream not supported", stream_options 400, or an idiosyncratic
        # 4xx). Retry non-streaming once; if the request itself is bad the
        # plain call reproduces the real error for the normal except-chains.
        logger.debug(
            "Auxiliary %s: streamed request failed (%s); retrying "
            "non-streaming", task or "call", exc,
        )
        return client.chat.completions.create(**kwargs)

    # Some shims (MoA virtual provider under quiet mode, defensive adapters)
    # return a complete response even when stream=True was requested.
    if hasattr(chunks, "choices"):
        _notify_aux_progress()
        return chunks
    return _aggregate_chat_stream(
        chunks, model=str(kwargs.get("model") or ""), total_ceiling=total_ceiling,
    )


def _aggregate_chat_stream(
    chunks: Any,
    *,
    model: str = "",
    total_ceiling: Optional[float] = None,
) -> Any:
    """Consume a chat.completions chunk stream into a complete response.

    Ticks the thread-local aux progress hook on every chunk. Raises
    TimeoutError when *total_ceiling* seconds elapse before the stream
    finishes — phrased with "timed out" so existing timeout classification
    (``_is_timeout_error``) treats it exactly like a request timeout.
    Accumulation is shared with the async mirror via
    :class:`_ChatStreamAccumulator`.
    """
    acc = _ChatStreamAccumulator(model=model, total_ceiling=total_ceiling)
    try:
        for chunk in chunks:
            acc.feed(chunk)
    finally:
        close_fn = getattr(chunks, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
    return acc.finish()


class _ChatStreamAccumulator:
    """Shared per-chunk accumulation for sync and async stream aggregation.

    Mirrors :func:`_aggregate_chat_stream`'s chunk handling so the async
    consumer below cannot drift from the sync one (same content/reasoning/
    tool-call delta reassembly, same "timed out" ceiling phrasing).
    """

    def __init__(self, model: str = "", total_ceiling: Optional[float] = None):
        self._started = time.monotonic()
        self._total_ceiling = total_ceiling
        self.content_parts: List[str] = []
        self.reasoning_parts: List[str] = []
        self.tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        self.finish_reason = None
        self.usage = None
        self.resp_id = ""
        self.resp_model = model or ""

    def feed(self, chunk: Any) -> None:
        _notify_aux_progress()
        if (
            self._total_ceiling is not None
            and (time.monotonic() - self._started) >= self._total_ceiling
        ):
            raise TimeoutError(
                f"Auxiliary streamed call timed out after {self._total_ceiling:.0f}s "
                "total ceiling (stream still open but over budget)"
            )
        self.resp_id = getattr(chunk, "id", None) or self.resp_id
        self.resp_model = getattr(chunk, "model", None) or self.resp_model
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage:
            self.usage = chunk_usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return
        choice = choices[0]
        self.finish_reason = getattr(choice, "finish_reason", None) or self.finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            return
        piece = getattr(delta, "content", None)
        if piece:
            self.content_parts.append(piece)
        reasoning_piece = (
            getattr(delta, "reasoning", None)
            or getattr(delta, "reasoning_content", None)
        )
        if reasoning_piece and isinstance(reasoning_piece, str):
            self.reasoning_parts.append(reasoning_piece)
        for tc in (getattr(delta, "tool_calls", None) or []):
            idx = getattr(tc, "index", 0) or 0
            acc = self.tool_calls_acc.setdefault(
                idx, {"id": "", "name": "", "arguments": []}
            )
            if getattr(tc, "id", None):
                acc["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    acc["name"] = fn.name
                if getattr(fn, "arguments", None):
                    acc["arguments"].append(fn.arguments)

    def finish(self) -> Any:
        tool_calls = None
        if self.tool_calls_acc:
            tool_calls = [
                SimpleNamespace(
                    id=acc["id"],
                    type="function",
                    function=SimpleNamespace(
                        name=acc["name"],
                        arguments="".join(acc["arguments"]),
                    ),
                )
                for _idx, acc in sorted(self.tool_calls_acc.items())
            ]
        message = SimpleNamespace(
            role="assistant",
            content="".join(self.content_parts),
            tool_calls=tool_calls,
            reasoning="".join(self.reasoning_parts) or None,
        )
        choice = SimpleNamespace(
            index=0,
            message=message,
            finish_reason=self.finish_reason or "stop",
        )
        return SimpleNamespace(
            id=self.resp_id,
            model=self.resp_model,
            object="chat.completion",
            choices=[choice],
            usage=self.usage,
        )


async def _aggregate_chat_stream_async(
    chunks: Any,
    *,
    model: str = "",
    total_ceiling: Optional[float] = None,
) -> Any:
    """Async mirror of :func:`_aggregate_chat_stream` (``async for`` consumer).

    The AsyncOpenAI stream contract is an async iterator — consuming it with
    the sync helper raises. Same accumulation and ceiling semantics via
    :class:`_ChatStreamAccumulator`.
    """
    acc = _ChatStreamAccumulator(model=model, total_ceiling=total_ceiling)
    try:
        async for chunk in chunks:
            acc.feed(chunk)
    finally:
        close_fn = getattr(chunks, "close", None) or getattr(chunks, "aclose", None)
        if callable(close_fn):
            try:
                result = close_fn()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
    return acc.finish()


async def _acreate_with_stream(
    client: Any,
    kwargs: Dict[str, Any],
    task: Optional[str] = None,
) -> Any:
    """Async chat.completions.create() for stream-only providers.

    Sends ``stream=True`` and aggregates the async chunk stream into a
    complete response (credit @kudi88, — async contract fixed to
    ``async for`` and tool-call deltas preserved per sweeper review).
    """
    total_ceiling = _aux_stream_total_ceiling(kwargs.get("timeout"))
    stream_kwargs = dict(kwargs)
    stream_kwargs["stream"] = True
    stream_kwargs["stream_options"] = {"include_usage": True}
    chunks = await client.chat.completions.create(**stream_kwargs)
    # Defensive: shims may hand back a complete response despite stream=True.
    if hasattr(chunks, "choices"):
        return chunks
    return await _aggregate_chat_stream_async(
        chunks, model=str(kwargs.get("model") or ""), total_ceiling=total_ceiling,
    )


@_relay_auxiliary_call
def call_llm(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = None,
    extra_body: dict = None,
    reasoning_config: Optional[dict] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    api_mode: str = None,
    stream: bool = False,
    stream_options: dict = None,
    route_info: Optional[Dict[str, str]] = None,
) -> Any:
    """Run an auxiliary LLM request, applying the configured task limit."""
    semaphore = _acquire_sync_aux_semaphore(task)
    if semaphore is not None:
        semaphore.acquire()
    try:
        response = _call_llm_impl(
            task=task,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            main_runtime=main_runtime,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            timeout=timeout,
            extra_body=extra_body,
            reasoning_config=reasoning_config,
            extra_headers=extra_headers,
            api_mode=api_mode,
            stream=stream,
            stream_options=stream_options,
            route_info=route_info,
        )
        if stream and semaphore is not None:
            stream_semaphore = semaphore
            semaphore = None
            return _release_sync_semaphore_after_stream(response, stream_semaphore)
        return response
    finally:
        if semaphore is not None:
            semaphore.release()


def _release_sync_semaphore_after_stream(
    stream: Any, semaphore: threading.BoundedSemaphore,
):
    """Release a permit only after a streaming response is consumed or closed."""
    try:
        yield from stream
    finally:
        try:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        finally:
            semaphore.release()


def _call_llm_impl(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = None,
    extra_body: dict = None,
    reasoning_config: Optional[dict] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    api_mode: str = None,
    stream: bool = False,
    stream_options: dict = None,
    route_info: Optional[Dict[str, str]] = None,
) -> Any:
    """Centralized synchronous LLM call.

    Resolves provider + model (from task config, explicit args, or auto-detect),
    handles auth, request formatting, and model-specific arg adjustments.

    Args:
        task: Auxiliary task name ("compression", "vision", "web_extract",
              "session_search", "skills_hub", "mcp", "title_generation").
              Reads provider:model from config/env. Ignored if provider is set.
        provider: Explicit provider override.
        model: Explicit model override.
        api_mode: Explicit API mode override (e.g. "codex_responses").
              Takes precedence over task config.
        messages: Chat messages list.
        temperature: Sampling temperature (None = provider default).
        max_tokens: Max output tokens (handles max_tokens vs max_completion_tokens).
        tools: Tool definitions (for function calling).
        timeout: Request timeout in seconds (None = read from auxiliary.{task}.timeout config).
        extra_body: Additional request body fields.
        reasoning_config: Optional Pilotage reasoning config for direct model calls
              such as MoA reference/aggregator slots.
        extra_headers: Additional per-request HTTP headers. These override
            client-level defaults for providers that gate capabilities on
        stream: When True, return the raw SDK streaming iterator instead of a
            validated complete response. The caller is responsible for consuming
            chunks (and for any fallback). Used by the MoA aggregator so its
            output can stream to the user.
        stream_options: Passed through to the request when stream is True
            (e.g. {"include_usage": True}).

    Returns:
        Response object with .choices[0].message.content, OR — when stream=True —
        the raw streaming iterator from client.chat.completions.create().

    Raises:
        RuntimeError: If no provider is configured.
    """
    # Capture one immutable runtime snapshot for keying, resolution, retries,
    # and fallbacks. Reading ambient state independently in each phase lets a
    # concurrent /model switch produce a key for one runtime and a client for
    # another.
    main_runtime = _normalize_main_runtime(main_runtime)
    resolved_provider, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        task, provider, model, base_url, api_key)
    if api_mode:
        resolved_api_mode = api_mode
    effective_extra_body = _get_task_extra_body(task)
    effective_extra_body.update(extra_body or {})
    effective_provider = resolved_provider

    if task == "vision":
        effective_provider, client, final_model = resolve_vision_provider_client(
            provider=resolved_provider if resolved_provider != "auto" else provider,
            model=resolved_model or model,
            base_url=resolved_base_url or base_url,
            api_key=resolved_api_key or api_key,
            async_mode=False,
            main_runtime=main_runtime,
        )
        if client is None and resolved_provider != "auto" and not resolved_base_url:
            logger.warning(
                "Vision provider %s unavailable, falling back to auto vision backends",
                resolved_provider,
            )
            effective_provider, client, final_model = resolve_vision_provider_client(
                provider="auto",
                model=resolved_model,
                async_mode=False,
                main_runtime=main_runtime,
            )
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: pilotage setup"
            )
        resolved_provider = effective_provider or resolved_provider
    else:
        client, final_model = _get_cached_client(
            resolved_provider,
            resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
            main_runtime=main_runtime,
            task=task,
        )
        effective_provider = _effective_provider_for_client(
            client, resolved_provider,
        )
        if client is None:
            _explicit = (resolved_provider or "").strip().lower()
            if _explicit and _explicit not in {"auto", "custom"}:
                raise RuntimeError(
                        f"Provider '{_explicit}' is set in config.yaml but no API key "
                        f"was found. Set the {_explicit.upper()}_API_KEY environment "
                        f"variable, or switch to a different provider with `pilotage model`."
                    )
            # For auto/custom with no credentials, try the full auto chain
            # rather than hardcoding one provider.  Pass model=None so each
            # provider uses its own default — resolved_model may be a
            # vendor-prefixed slug that doesn't work elsewhere.
            if client is None and not resolved_base_url:
                logger.info("Auxiliary %s: provider %s unavailable, trying auto-detection chain",
                            task or "call", resolved_provider)
                client, final_model = _get_cached_client(
                    "auto", main_runtime=main_runtime, task=task,
                )
                effective_provider = _effective_provider_for_client(
                    client, "auto",
                )
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: pilotage setup")

    effective_timeout = _effective_aux_timeout(task, timeout)
    request_provider = effective_provider or resolved_provider
    _set_relay_auxiliary_route(
        request_provider,
        final_model,
        resolved_api_mode,
    )
    _record_route_info(route_info, request_provider, final_model)

    # Log what we're about to do — makes auxiliary operations visible
    _base_info = str(getattr(client, "base_url", resolved_base_url) or "")
    if task:
        logger.info("Auxiliary %s: using %s (%s)%s",
                     task, request_provider or "auto", final_model or "default",
                     f" at {_base_info}" if _base_info else "")

    # Pass the client's actual base_url (not just resolved_base_url) so
    # endpoint-specific temperature overrides can distinguish sibling
    # endpoints even on auto-detected routes.
    kwargs = _build_call_kwargs(
        request_provider, final_model, messages,
        temperature=temperature, max_tokens=max_tokens,
        tools=tools, timeout=effective_timeout, extra_body=effective_extra_body,
        reasoning_config=reasoning_config,
        base_url=_base_info or resolved_base_url, task=task)
    if extra_headers:
        kwargs["extra_headers"] = dict(extra_headers)

    _client_base = str(getattr(client, "base_url", "") or "")

    # Streaming path: return the raw SDK Stream iterator directly. This is used by
    # the MoA aggregator so its tokens stream to the user. It deliberately skips
    # _validate_llm_response and the temperature/max_tokens/payment fallback chain
    # below — those all assume a complete response object, whereas a stream is
    # consumed chunk-by-chunk by the caller. The caller (the agent's streaming
    # consumer) owns chunk reassembly, stale-stream detection, and falling back to
    # a non-streaming call on error. stream_options is best-effort: providers that
    # reject it surface an error the caller's fallback already handles.
    if stream:
        kwargs["stream"] = True
        if stream_options:
            kwargs["stream_options"] = stream_options
        return _relay_sync_stream(
            client,
            kwargs,
            provider=request_provider,
            api_mode=resolved_api_mode,
        )

    # Handle unsupported temperature, max_tokens vs max_completion_tokens retry,
    # then payment fallback.
    try:
        # Retry on the same provider for a transient transport blip
        # (connection reset / streaming-close / incomplete chunked read / 5xx /
        # 408) before the except-chain below escalates to provider/model
        # fallback. A dropped connection shouldn't abandon an otherwise-healthy
        # provider — this especially matters for pinned auxiliary calls like MoA
        # reference advisors, where "fallback to another provider" is not a
        # meaningful recovery (the advisor is a specific model), so a transient
        # blip that isn't retried simply loses that advisor for the turn (root
        # of the run2 double-advisor "Connection error" collapse — a genuine
        # upstream blip hitting both parallel advisors at once).
        #
        # Attempts are bounded and use exponential backoff. Count is configurable
        # via auxiliary.transient_retries (default 2 retries → 3 total attempts);
        # a second/third failure or any non-transient error falls through to
        # ``first_err`` and the existing fallback handling unchanged. Unified home
        # for the transient retry every auxiliary task shares. 
        try:
            return _validate_llm_response(
                _relay_sync_completion(
                    client,
                    kwargs,
                    provider=request_provider,
                    api_mode=resolved_api_mode,
                    create=lambda request: _create_with_progress(
                        client,
                        request,
                        task,
                        force_stream=_provider_requires_stream(
                            request_provider, _base_info or resolved_base_url,
                        ),
                    ),
                ),
                task,
                provider=request_provider, base_url=_base_info)
        except Exception as transient_err:
            if not _is_transient_transport_error(transient_err):
                raise
            # Compression is on the critical preflight path: a user cannot
            # continue or resume an oversized session until it compacts. A
            # same-provider retry on a timeout means another full ``timeout``-
            # long wall-clock block before the except-chain below can fall
            # back — doubling the user-visible stall. Skip the
            # same-provider retry for compression on a full-budget timeout and
            # fall straight through to provider/model fallback; fast blips (a
            # streaming-close or a 5xx) still retry, since those are cheap.
            if task == "compression" and _is_timeout_error(transient_err):
                logger.info(
                    "Auxiliary compression: timeout on the critical path; "
                    "skipping same-provider retry and falling back: %s",
                    transient_err,
                )
                raise
            _max_transient_retries = _transient_retry_count()
            _last_transient = transient_err
            for _attempt in range(1, _max_transient_retries + 1):
                _backoff = min(_TRANSIENT_RETRY_BACKOFF_BASE * (2.0 ** (_attempt - 1)), 8.0)
                logger.info(
                    "Auxiliary %s: transient transport error (attempt %d/%d); "
                    "retrying same provider after %.1fs before fallback: %s",
                    task or "call", _attempt, _max_transient_retries, _backoff,
                    _last_transient,
                )
                time.sleep(_backoff)
                try:
                    return _validate_llm_response(
                        _relay_sync_completion(
                            client,
                            kwargs,
                            provider=request_provider,
                            api_mode=resolved_api_mode,
                            create=lambda request: _create_with_progress(
                                client,
                                request,
                                task,
                                force_stream=_provider_requires_stream(
                                    request_provider,
                                    _base_info or resolved_base_url,
                                ),
                            ),
                        ),
                        task)
                except Exception as retry_transient:
                    if not _is_transient_transport_error(retry_transient):
                        raise
                    _last_transient = retry_transient
            # Retries exhausted — fall through to first_err fallback handling.
            raise _last_transient
    except Exception as first_err:
        if "temperature" in kwargs and _is_unsupported_temperature_error(first_err):
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("temperature", None)
            logger.info(
                "Auxiliary %s: provider rejected temperature; retrying once without it",
                task or "call",
            )
            try:
                return _validate_llm_response(
                    _relay_sync_completion(
                        client,
                        retry_kwargs,
                        provider=resolved_provider,
                        api_mode=resolved_api_mode,
                    ), task)
            except Exception as retry_err:
                retry_err_str = str(retry_err)
                # If retry still fails, fall through to the max_tokens /
                # payment / auth chains below using the temperature-stripped
                # kwargs.  Re-raise only if the retry hit something those
                # chains won't handle.
                if not (
                    _is_payment_error(retry_err)
                    or _is_connection_error(retry_err)
                    or _is_auth_error(retry_err)
                    or "max_tokens" in retry_err_str
                    or "unsupported_parameter" in retry_err_str
                ):
                    raise
                first_err = retry_err
                kwargs = retry_kwargs

        err_str = str(first_err)
        if max_tokens is not None and (
            "max_tokens" in err_str
            or "unsupported_parameter" in err_str
            or _is_unsupported_parameter_error(first_err, "max_tokens")
        ):
            kwargs.pop("max_tokens", None)
            kwargs.pop("max_completion_tokens", None)
            try:
                return _validate_llm_response(
                    _relay_sync_completion(
                        client,
                        kwargs,
                        provider=resolved_provider,
                        api_mode=resolved_api_mode,
                    ), task)
            except Exception as retry_err:
                # If the max_tokens retry also hits a payment or connection
                # error, fall through to the fallback chain below.
                if not (_is_payment_error(retry_err) or _is_connection_error(retry_err) or _is_rate_limit_error(retry_err)):
                    raise
                first_err = retry_err

        # ── Auth refresh retry ───────────────────────────────────────
        auth_refresh_provider = _auth_refresh_provider_for_route(
            resolved_provider, _base_info)
        if (_is_auth_error(first_err)
                and auth_refresh_provider not in {"auto", "", None}):
            if _refresh_provider_credentials(auth_refresh_provider):
                if auth_refresh_provider != _normalize_aux_provider(resolved_provider):
                    # The stale client is cached under the route label
                    # (e.g. "auto"), not the concrete backend we refreshed.
                    _evict_cached_clients(resolved_provider)
                logger.info(
                    "Auxiliary %s: refreshed %s credentials after auth error, retrying",
                    task or "call", auth_refresh_provider,
                )
                return _retry_same_provider_sync(
                    task=task,
                    resolved_provider=auth_refresh_provider,
                    resolved_model=resolved_model or final_model,
                    resolved_base_url=resolved_base_url,
                    resolved_api_key=resolved_api_key,
                    resolved_api_mode=resolved_api_mode,
                    main_runtime=main_runtime,
                    final_model=final_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    effective_timeout=effective_timeout,
                    effective_extra_body=effective_extra_body,
                    reasoning_config=reasoning_config,
                    extra_headers=extra_headers,
                )

        # ── Same-provider credential-pool recovery ─────────────────────
        pool_provider = _recoverable_pool_provider(resolved_provider, client, main_runtime=main_runtime)
        # Capture the exact API key used so mark_exhausted_and_rotate can find
        # the correct pool entry even when another process rotated the pool
        # between this call and recovery (which leaves current()=None and makes
        # _select_unlocked() return the NEXT key by mistake).
        _client_api_key = str(getattr(client, "api_key", "") or "")
        if pool_provider and (_is_auth_error(first_err) or _is_payment_error(first_err) or _is_rate_limit_error(first_err)):
            recovery_err = first_err
            # Skip the extra retry for clear payment/quota errors — the endpoint
            # won't accept another request with the same exhausted key.
            if _is_rate_limit_error(first_err) and not _is_payment_error(first_err):
                try:
                    return _validate_llm_response(
                        _relay_sync_completion(
                            client,
                            kwargs,
                            provider=resolved_provider,
                            api_mode=resolved_api_mode,
                        ), task)
                except Exception as retry_err:
                    if not (_is_auth_error(retry_err) or _is_payment_error(retry_err) or _is_rate_limit_error(retry_err)):
                        raise
                    recovery_err = retry_err
            if _recover_provider_pool(pool_provider, recovery_err, failed_api_key=_client_api_key):
                logger.info(
                    "Auxiliary %s: recovered %s via credential-pool rotation after %s",
                    task or "call", pool_provider, type(recovery_err).__name__,
                )
                try:
                    return _retry_same_provider_sync(
                        task=task,
                        resolved_provider=resolved_provider,
                        resolved_model=resolved_model,
                        resolved_base_url=resolved_base_url,
                        resolved_api_key=resolved_api_key,
                        resolved_api_mode=resolved_api_mode,
                        main_runtime=main_runtime,
                        final_model=final_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        effective_timeout=effective_timeout,
                        effective_extra_body=effective_extra_body,
                        reasoning_config=reasoning_config,
                        extra_headers=extra_headers,
                    )
                except Exception as retry2_err:
                    # The rotated key also hit a quota/auth wall.  Mark it
                    # immediately so concurrent processes don't make a
                    # redundant API call to discover it's exhausted too.
                    # Then fall through to the payment fallback below so
                    # alternative providers can still serve the request.
                    if (_is_payment_error(retry2_err) or _is_auth_error(retry2_err)
                            or _is_rate_limit_error(retry2_err)):
                        _recover_provider_pool(pool_provider, retry2_err)
                        first_err = retry2_err
                    else:
                        raise

        # Connection/timeout errors leave the cached client poisoned (closed
        # httpx transport, half-read stream, dead async loop).  Drop it from
        # the cache regardless of whether we found a fallback above so the
        # next auxiliary call rebuilds a fresh client instead of reusing the
        # dead one. See.
        if _is_connection_error(first_err):
            try:
                _evict_cached_client_instance(client)
            except Exception:
                logger.debug("Auxiliary: cache eviction after connection error failed",
                             exc_info=True)
        raise


def extract_content_or_reasoning(response) -> str:
    """Extract content from an LLM response, falling back to reasoning fields.

    Mirrors the main agent loop's behavior when a reasoning model returns
    ``content=None`` with reasoning in structured fields.

    Resolution order:
      1. ``message.content`` — strip inline think/reasoning blocks, check for
         remaining non-whitespace text.
      2. ``message.reasoning`` / ``message.reasoning_content`` — direct
         structured reasoning fields.
      3. ``message.reasoning_details`` — unified array format.

    Returns the best available text, or ``""`` if nothing found.
    """
    import re

    msg = response.choices[0].message
    content = (msg.content or "").strip()

    if content:
        # Strip inline think/reasoning blocks (mirrors _strip_think_blocks)
        cleaned = re.sub(
            r"<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>"
            r".*?"
            r"</(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>",
            "", content, flags=re.DOTALL | re.IGNORECASE,
        ).strip()
        if cleaned:
            return cleaned

    # Content is empty or reasoning-only — try structured reasoning fields
    reasoning_parts: list[str] = []
    for field in ("reasoning", "reasoning_content"):
        val = getattr(msg, field, None)
        if val and isinstance(val, str) and val.strip() and val not in reasoning_parts:
            reasoning_parts.append(val.strip())

    details = getattr(msg, "reasoning_details", None)
    if details and isinstance(details, list):
        for detail in details:
            if isinstance(detail, dict):
                summary = (
                    detail.get("summary")
                    or detail.get("content")
                    or detail.get("text")
                )
                if summary and summary not in reasoning_parts:
                    reasoning_parts.append(summary.strip() if isinstance(summary, str) else str(summary))

    if reasoning_parts:
        return "\n\n".join(reasoning_parts)

    return ""


@_relay_auxiliary_call_async
async def async_call_llm(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = None,
    extra_body: dict = None,
    reasoning_config: Optional[dict] = None,
    route_info: Optional[Dict[str, str]] = None,
) -> Any:
    """Run an asynchronous auxiliary LLM request under the configured limit."""
    semaphore = _acquire_async_aux_semaphore(task)
    if semaphore is not None:
        await semaphore.acquire()
    try:
        return await _async_call_llm_impl(
            task=task,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            main_runtime=main_runtime,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            timeout=timeout,
            extra_body=extra_body,
            reasoning_config=reasoning_config,
            route_info=route_info,
        )
    finally:
        if semaphore is not None:
            semaphore.release()


async def _async_call_llm_impl(
    task: str = None,
    *,
    provider: str = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    messages: list,
    temperature: Optional[float] = None,
    max_tokens: int = None,
    tools: list = None,
    timeout: float = None,
    extra_body: dict = None,
    reasoning_config: Optional[dict] = None,
    route_info: Optional[Dict[str, str]] = None,
) -> Any:
    """Centralized asynchronous LLM call.

    Same as call_llm() but async. See call_llm() for full documentation.
    """
    # Keep every async phase on the same runtime identity, even if another
    # session switches models while this task is awaiting network I/O.
    main_runtime = _normalize_main_runtime(main_runtime)
    resolved_provider, resolved_model, resolved_base_url, resolved_api_key, resolved_api_mode = _resolve_task_provider_model(
        task, provider, model, base_url, api_key)
    effective_extra_body = _get_task_extra_body(task)
    effective_extra_body.update(extra_body or {})
    effective_provider = resolved_provider

    if task == "vision":
        effective_provider, client, final_model = resolve_vision_provider_client(
            provider=resolved_provider if resolved_provider != "auto" else provider,
            model=resolved_model or model,
            base_url=resolved_base_url or base_url,
            api_key=resolved_api_key or api_key,
            async_mode=True,
            main_runtime=main_runtime,
        )
        if client is None and resolved_provider != "auto" and not resolved_base_url:
            logger.warning(
                "Vision provider %s unavailable, falling back to auto vision backends",
                resolved_provider,
            )
            effective_provider, client, final_model = resolve_vision_provider_client(
                provider="auto",
                model=resolved_model,
                async_mode=True,
                main_runtime=main_runtime,
            )
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: pilotage setup"
            )
        resolved_provider = effective_provider or resolved_provider
    else:
        client, final_model = _get_cached_client(
            resolved_provider,
            resolved_model,
            async_mode=True,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            api_mode=resolved_api_mode,
            main_runtime=main_runtime,
            task=task,
        )
        effective_provider = _effective_provider_for_client(
            client, resolved_provider,
        )
        if client is None:
            _explicit = (resolved_provider or "").strip().lower()
            if _explicit and _explicit not in {"auto", "custom"}:
                raise RuntimeError(
                        f"Provider '{_explicit}' is set in config.yaml but no API key "
                        f"was found. Set the {_explicit.upper()}_API_KEY environment "
                        f"variable, or switch to a different provider with `pilotage model`."
                    )
            if client is None and not resolved_base_url:
                logger.info("Auxiliary %s: provider %s unavailable, trying auto-detection chain",
                            task or "call", resolved_provider)
                client, final_model = _get_cached_client(
                    "auto",
                    async_mode=True,
                    main_runtime=main_runtime,
                    task=task,
                )
                effective_provider = _effective_provider_for_client(
                    client, "auto",
                )
        if client is None:
            raise RuntimeError(
                f"No LLM provider configured for task={task} provider={resolved_provider}. "
                f"Run: pilotage setup")

    effective_timeout = _effective_aux_timeout(task, timeout)
    request_provider = effective_provider or resolved_provider
    _set_relay_auxiliary_route(
        request_provider,
        final_model,
        resolved_api_mode,
    )
    _record_route_info(route_info, request_provider, final_model)

    # Pass the client's actual base_url (not just resolved_base_url) so
    # endpoint-specific temperature overrides can distinguish sibling
    # endpoints even on auto-detected routes.
    _client_base = str(getattr(client, "base_url", "") or "")
    kwargs = _build_call_kwargs(
        request_provider, final_model, messages,
        temperature=temperature, max_tokens=max_tokens,
        tools=tools, timeout=effective_timeout, extra_body=effective_extra_body,
        reasoning_config=reasoning_config,
        base_url=_client_base or resolved_base_url, task=task)


    try:
        # Retry ONCE on the same provider for a transient transport blip
        # before the except-chain escalates to fallback — see call_llm()
        # for the rationale. 
        _force_stream_async = (
            _provider_requires_stream(
                request_provider, _client_base or resolved_base_url,
            )
            and not isinstance(client, AsyncCodexAuxiliaryClient)
        )

        async def _acreate(_kwargs: Dict[str, Any]) -> Any:
            if _force_stream_async:
                return await _acreate_with_stream(client, _kwargs, task)
            return await client.chat.completions.create(**_kwargs)

        try:
            return _validate_llm_response(
                await _relay_async_completion(
                    client,
                    kwargs,
                    provider=request_provider,
                    api_mode=resolved_api_mode,
                    create=_acreate,
                ),
                task,
                provider=request_provider, base_url=_client_base)
        except Exception as transient_err:
            if not _is_transient_transport_error(transient_err):
                raise
            # See call_llm(): compression is on the critical preflight path,
            # so skip the same-provider retry on a full-budget timeout and
            # fall straight through to fallback.
            if task == "compression" and _is_timeout_error(transient_err):
                logger.info(
                    "Auxiliary compression (async): timeout on the critical "
                    "path; skipping same-provider retry and falling back: %s",
                    transient_err,
                )
                raise
            logger.info(
                "Auxiliary %s (async): transient transport error; retrying "
                "once on the same provider before fallback: %s",
                task or "call", transient_err,
            )
            return _validate_llm_response(
                await _relay_async_completion(
                    client,
                    kwargs,
                    provider=request_provider,
                    api_mode=resolved_api_mode,
                    create=_acreate,
                ),
                task)
    except Exception as first_err:
        if "temperature" in kwargs and _is_unsupported_temperature_error(first_err):
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("temperature", None)
            logger.info(
                "Auxiliary %s (async): provider rejected temperature; retrying once without it",
                task or "call",
            )
            try:
                return _validate_llm_response(
                    await _relay_async_completion(
                        client,
                        retry_kwargs,
                        provider=resolved_provider,
                        api_mode=resolved_api_mode,
                    ), task)
            except Exception as retry_err:
                retry_err_str = str(retry_err)
                if not (
                    _is_payment_error(retry_err)
                    or _is_connection_error(retry_err)
                    or _is_auth_error(retry_err)
                    or "max_tokens" in retry_err_str
                    or "unsupported_parameter" in retry_err_str
                ):
                    raise
                first_err = retry_err
                kwargs = retry_kwargs

        err_str = str(first_err)
        if max_tokens is not None and (
            "max_tokens" in err_str
            or "unsupported_parameter" in err_str
            or _is_unsupported_parameter_error(first_err, "max_tokens")
        ):
            kwargs.pop("max_tokens", None)
            kwargs.pop("max_completion_tokens", None)
            try:
                return _validate_llm_response(
                    await _relay_async_completion(
                        client,
                        kwargs,
                        provider=resolved_provider,
                        api_mode=resolved_api_mode,
                    ), task)
            except Exception as retry_err:
                # If the max_tokens retry also hits a payment or connection
                # error, fall through to the fallback chain below.
                if not (_is_payment_error(retry_err) or _is_connection_error(retry_err) or _is_rate_limit_error(retry_err)):
                    raise
                first_err = retry_err

        # ── Auth refresh retry (mirrors sync call_llm) ───────────────
        auth_refresh_provider = _auth_refresh_provider_for_route(
            resolved_provider, _client_base)
        if (_is_auth_error(first_err)
                and auth_refresh_provider not in {"auto", "", None}):
            if _refresh_provider_credentials(auth_refresh_provider):
                if auth_refresh_provider != _normalize_aux_provider(resolved_provider):
                    # The stale client is cached under the route label
                    # (e.g. "auto"), not the concrete backend we refreshed.
                    _evict_cached_clients(resolved_provider)
                logger.info(
                    "Auxiliary %s (async): refreshed %s credentials after auth error, retrying",
                    task or "call", auth_refresh_provider,
                )
                return await _retry_same_provider_async(
                    task=task,
                    resolved_provider=auth_refresh_provider,
                    resolved_model=resolved_model or final_model,
                    resolved_base_url=resolved_base_url,
                    resolved_api_key=resolved_api_key,
                    resolved_api_mode=resolved_api_mode,
                    final_model=final_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    effective_timeout=effective_timeout,
                    effective_extra_body=effective_extra_body,
                    reasoning_config=reasoning_config,
                )

        # ── Same-provider credential-pool recovery (mirrors sync) ─────
        pool_provider = _recoverable_pool_provider(resolved_provider, client, main_runtime=main_runtime)
        _client_api_key = str(getattr(client, "api_key", "") or "")
        if pool_provider and (_is_auth_error(first_err) or _is_payment_error(first_err) or _is_rate_limit_error(first_err)):
            recovery_err = first_err
            # Skip the extra retry for clear payment/quota errors — the endpoint
            # won't accept another request with the same exhausted key.
            if _is_rate_limit_error(first_err) and not _is_payment_error(first_err):
                try:
                    return _validate_llm_response(
                        await _relay_async_completion(
                            client,
                            kwargs,
                            provider=resolved_provider,
                            api_mode=resolved_api_mode,
                        ), task)
                except Exception as retry_err:
                    if not (_is_auth_error(retry_err) or _is_payment_error(retry_err) or _is_rate_limit_error(retry_err)):
                        raise
                    recovery_err = retry_err
            if _recover_provider_pool(pool_provider, recovery_err, failed_api_key=_client_api_key):
                logger.info(
                    "Auxiliary %s (async): recovered %s via credential-pool rotation after %s",
                    task or "call", pool_provider, type(recovery_err).__name__,
                )
                try:
                    return await _retry_same_provider_async(
                        task=task,
                        resolved_provider=resolved_provider,
                        resolved_model=resolved_model,
                        resolved_base_url=resolved_base_url,
                        resolved_api_key=resolved_api_key,
                        resolved_api_mode=resolved_api_mode,
                        final_model=final_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        effective_timeout=effective_timeout,
                        effective_extra_body=effective_extra_body,
                        reasoning_config=reasoning_config,
                    )
                except Exception as retry2_err:
                    if (_is_payment_error(retry2_err) or _is_auth_error(retry2_err)
                            or _is_rate_limit_error(retry2_err)):
                        _recover_provider_pool(pool_provider, retry2_err)
                        first_err = retry2_err
                    else:
                        raise

        # Mirror the sync path: drop poisoned clients on connection/timeout
        # so the next aux call rebuilds. See.
        if _is_connection_error(first_err):
            try:
                _evict_cached_client_instance(client)
            except Exception:
                logger.debug("Auxiliary (async): cache eviction after connection error failed",
                             exc_info=True)
        raise
