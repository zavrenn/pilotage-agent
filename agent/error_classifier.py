"""API error classification for smart failover and recovery.

Provides a structured taxonomy of API errors and a priority-ordered
classification pipeline that determines the correct recovery action
(retry, rotate credential, fallback to another provider, compress
context, or abort).

Replaces scattered inline string-matching with a centralized classifier
that the main retry loop in run_agent.py consults for every API failure.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Synthetic error code used when the OpenAI SDK rejects a provider's SSE
# ``data:`` field before Pilotage receives a completion chunk.  Keeping this
# distinct from generic JSON parse failures lets the classifier make narrow,
# provider-stream-specific recovery decisions without inventing an HTTP status.
PROVIDER_STREAM_NON_JSON_ERROR_CODE = "provider_stream_non_json_data"


# ── Error taxonomy ──────────────────────────────────────────────────────

class FailoverReason(enum.Enum):
    """Why an API call failed — determines recovery strategy."""

    # Authentication / authorization
    auth = "auth"                        # Transient auth (401/403) — refresh/rotate
    auth_permanent = "auth_permanent"    # Auth failed after refresh — abort

    # Billing / quota
    billing = "billing"                  # 402 or confirmed credit exhaustion — rotate immediately
    rate_limit = "rate_limit"            # 429 or quota-based throttling — backoff then rotate
    # Upstream model rate-limited (aggregator 429) — fallback to a different
    # model, NOT credential rotation. The user's key is healthy.

    # Server-side
    overloaded = "overloaded"            # 503/529 — provider overloaded, backoff
    server_error = "server_error"        # 500/502 — internal server error, retry

    # Transport
    timeout = "timeout"                  # Connection/read timeout — rebuild client + retry
    # TLS certificate verification failure — deterministic for the host
    # (TLS-inspecting proxy, missing/expired CA bundle, self-signed cert).
    # Retrying reproduces the identical handshake failure, so fail fast
    # with actionable guidance instead of burning retries.
    ssl_cert_verification = "ssl_cert_verification"

    # Context / payload
    context_overflow = "context_overflow"  # Context too large — compress, not failover
    payload_too_large = "payload_too_large"  # 413 — compress payload
    image_too_large = "image_too_large"   # Native image part exceeds provider's per-image limit — shrink and retry

    # Model / provider policy
    model_not_found = "model_not_found"  # 404 or invalid model — fallback to different model
    content_policy_blocked = "content_policy_blocked"  # Provider safety filter rejected this prompt — deterministic per-request, don't retry unchanged

    # Request format
    format_error = "format_error"        # 400 bad request — abort or strip + retry
    invalid_encrypted_content = "invalid_encrypted_content"  # Responses replay blob rejected — strip replay state and retry
    multimodal_tool_content_unsupported = "multimodal_tool_content_unsupported"  # Provider rejected list-type content in tool messages (e.g. Xiaomi MiMo) — downgrade to text and retry

    # Catch-all
    unknown = "unknown"                  # Unclassifiable — retry with backoff


# ── Classification result ───────────────────────────────────────────────

@dataclass
class ClassifiedError:
    """Structured classification of an API error with recovery hints."""

    reason: FailoverReason
    status_code: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    message: str = ""
    error_context: Dict[str, Any] = field(default_factory=dict)

    # Recovery action hints — the retry loop checks these instead of
    # re-classifying the error itself.
    retryable: bool = True
    should_compress: bool = False
    should_rotate_credential: bool = False
    should_fallback: bool = False

    @property
    def is_auth(self) -> bool:
        return self.reason in {FailoverReason.auth, FailoverReason.auth_permanent}



# ── Provider-specific patterns ──────────────────────────────────────────

# Patterns that indicate billing exhaustion (not transient rate limit)
_BILLING_PATTERNS = [
    "insufficient credits",
    "insufficient_quota",
    "insufficient balance",
    "credit balance",
    "credits exhausted",
    "credits have been exhausted",
    "requires available credits",
    "account balance is too low",
    "no usable credits",
    "top up your credits",
    "payment required",
    "billing hard limit",
    "exceeded your current quota",
    "account is deactivated",
    "plan does not include",
    "out of funds",
    "run out of funds",
    "balance_depleted",
    "model_not_supported_on_free_tier",
    "not available on the free tier",
]

# Structured provider codes that mean the account cannot serve paid traffic
# until credits/subscription capacity is restored.
_BILLING_ERROR_CODES = frozenset({
    "insufficient_quota",
    "billing_not_active",
    "payment_required",
    "insufficient_credits",
    "no_usable_credits",
    "balance_depleted",
    "model_not_supported_on_free_tier",
    "member_spend_cap_exceeded",
})

# Patterns that indicate rate limiting (transient, will resolve)
_RATE_LIMIT_PATTERNS = [
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttled",
    "requests per minute",
    "tokens per minute",
    "requests per day",
    "try again in",
    "please retry after",
    "resource_exhausted",
    "throttlingexception",
    # Generic throttle prefix — some proxies surface throttling as
    # "Throttling error: Too many tokens, please wait before trying again."
    # Without this entry the message falls through to the context-overflow
    # list (which contains "too many tokens") and the retry loop compresses
    # a healthy session instead of backing off.  Matched BEFORE
    # _CONTEXT_OVERFLOW_PATTERNS in the message-only path, so throttle wins.
    "throttling",
]

# Patterns that indicate provider-side overload, NOT a per-credential rate
# limit or billing problem.  The credential is valid — the server is just
# busy — so the correct recovery is "back off and retry the same key", never
# "rotate the credential" (rotating exhausts the pool while the endpoint is
# still busy; a single-key user has nothing to rotate to).  Some providers
# Some providers reuse HTTP 429 for server-wide overload, so the 429
# status path matches the body against this list before falling through to
# the rate_limit default.  Phrases are kept narrow and overload-flavoured so a
# normal rate-limit message ("you have been rate-limited") doesn't hit this
# bucket.
_OVERLOADED_PATTERNS = [
    "overloaded",
    "temporarily overloaded",
    "service is temporarily overloaded",
    "service may be temporarily overloaded",
    "server is overloaded",
    "server overloaded",
    "service overloaded",
    "service is overloaded",
    "upstream overloaded",
    "currently overloaded",
    "at capacity",
    "over capacity",
]

# Usage-limit patterns that need disambiguation (could be billing OR rate_limit)
_USAGE_LIMIT_PATTERNS = [
    "usage limit",
    "quota",
    "limit exceeded",
    "key limit exceeded",
]

# Patterns confirming usage limit is transient (not billing)
_USAGE_LIMIT_TRANSIENT_SIGNALS = [
    "try again",
    "retry",
    "resets at",
    "reset in",
    "wait",
    "requests remaining",
    "periodic",
    "window",
]

# Payload-too-large patterns detected from message text (no status_code attr).
# Proxies and some backends embed the HTTP status in the error message.
_PAYLOAD_TOO_LARGE_PATTERNS = [
    "request entity too large",
    "payload too large",
    "error code: 413",
    # Structured 413 error type.  Normally arrives with an HTTP 413 status
    # (handled by the status path), but proxies can re-wrap it into a plain
    # message with no status attribute — route it to the same recovery.
    "request_too_large",
    "request exceeds the maximum size",
]

# Image-size patterns.  Matched against 400 bodies (not 413) because most
# providers return a 400 with a specific image-too-big message before the
# whole request hits the 413 size limit.
_IMAGE_TOO_LARGE_PATTERNS = [
    "image exceeds",
    "image too large",
    "image_too_large",
    "image size exceeds",
    "image dimensions exceed",
    "dimensions exceed max allowed size",
    # "request_too_large" on a request known to contain an image → image is
    # the likely culprit; we still try the shrink path before giving up.
]

# Providers that follow the OpenAI spec strictly require tool message
# ``content`` to be a string.  OpenAI and the Codex Responses API extend this
# to accept a content-parts list (text + image_url) so screenshots from
# computer_use survive.  Stricter OpenAI-compatible endpoints reject the list
# with a 400 — the patterns below are the most common error shapes.
# Recovery: strip image parts from tool messages in-place, record the
# (provider, model) for the rest of the session, retry.
_MULTIMODAL_TOOL_CONTENT_PATTERNS = [
    "text is not set",
    # Generic "tool message must be string" shapes
    "tool message content must be a string",
    "tool content must be a string",
    "tool message must be a string",
    # OpenAI-compat servers that reject list-type tool content with a
    # schema-validation message
    "expected string, got list",
    "expected string, got array",
    "tool_call.content must be string",
]

# Context overflow patterns
_CONTEXT_OVERFLOW_PATTERNS = [
    "context length",
    "context size",
    "maximum context",
    "token limit",
    "too many tokens",
    "reduce the length",
    "exceeds the limit",
    "context window",
    "prompt is too long",
    "prompt exceeds max length",
    # NOTE: bare "max_tokens" is load-bearing — the output-cap-retry path keys
    # off it (e.g. "max_tokens: 65536 > context_window: 200000 ..."). Do NOT
    # remove it. Provider empty-response advisories also contain "very low
    # max_tokens", but those are intercepted by _EMPTY_PROVIDER_RESPONSE_PATTERNS
    # BEFORE this list is consulted, so they never mis-route into compression.
    "max_tokens",
    "maximum number of tokens",
    "input is too long",
    "context length exceeded",
    # Chinese error messages (some providers return these)
    "超过最大长度",
    "上下文长度",
    "max input token",
    "input token",
    "exceeds the maximum number of input tokens",
    # "Input length 131393 exceeds the maximum allowed input length of
    # 131040 tokens." — no other pattern in this list matches that wording.
    "maximum allowed input length",
]

# Model not found patterns
_MODEL_NOT_FOUND_PATTERNS = [
    "is not a valid model",
    "invalid model",
    "model not found",
    "model_not_found",
    "does not exist",
    "no such model",
    "unknown model",
    "unsupported model",
]


# Malformed-message-array 400s.  Deterministic request-shape rejections that
# describe the *transcript* being invalid, not a parameter.  The canonical
# case: a stream dies mid-response and Pilotage persists a content-less
# assistant stub; on the next turn the provider's message schema rejects the
# whole request with
#   "all messages must have non-empty content except for the optional final
#    assistant message"  /  errorCode INVALID_REQUEST_BODY
# These are NOT context overflow — the input may be tiny — but a large
# session used to mis-route them into the compression loop via the generic
# "400 + large session" heuristic below, ending in "Cannot compress further"
# every retry (the input is unchanged, so compression cannot help).  Match
# the message-shape signals explicitly and fail fast as a format_error so the
# loop stops looping.  The empty-stub creation is the root cause (fixed in
# chat_completion_helpers); this pattern stops the misclassification symptom
# for transcripts that already contain a poisoned stub.
# Chat-template raise_exception("No user query found in messages").
_NO_USER_QUERY_SIGNAL = "no user query found"

_INVALID_MESSAGE_BODY_PATTERNS = [
    "must have non-empty content",
    "messages must have non-empty",
    "invalid_request_body",
    "text content blocks must be non-empty",
    "content field is required",
    "messages: at least one message is required",
    # Chat templates raise this when the request has no surviving non-empty
    # user turn (oversized session truncation, compression that dropped the
    # only user message, or a resumed lineage that opens with assistant or
    # tool). Deterministic — compression cannot invent a user query the
    # template already rejected, so fail fast as format_error.
    _NO_USER_QUERY_SIGNAL,
]

# Request-validation patterns — the request is malformed and will fail
# identically on every retry. Some OpenAI-compatible gateways (notably
# codex.nekos.me) return these as 5xx instead of the standard 4xx, which
# makes the generic "5xx → retryable server_error" rule misfire: the retry
# loop hammers the same deterministic rejection 3+ times, then the
# transport-recovery path resets the counter and does it again, producing
# a request flood. When a 5xx body carries one of these unambiguous
# request-validation signals, classify as a non-retryable format_error so
# the loop fails fast and falls back instead of looping.
_REQUEST_VALIDATION_PATTERNS = [
    "unknown parameter",
    "unsupported parameter",
    "unrecognized request argument",
    "invalid_request_error",
    "unknown_parameter",
    "unsupported_parameter",
]

# Provider content-policy / safety-filter blocks — *per-prompt* safety
# decisions made by the model provider. They are deterministic for the unchanged
# request, so retrying the same prompt three times just reproduces the same
# block and burns paid attempts on a refusal. The recovery is to switch to a
# configured fallback model/provider immediately, or surface the block to
# the user with actionable guidance if no fallback exists.
#
# Patterns are intentionally narrow — each phrase is a verbatim string from
# a specific provider's safety pipeline, not a generic word like "policy" or
# "violation" that could collide with billing/auth/format errors:
#   • OpenAI Codex cybersecurity refusal
#   • OpenAI moderation refusal ("violates our usage policies", with
#     "usage policies" disambiguating from billing's "exceeded ... policy")
#   • OpenAI Responses content filter
_CONTENT_POLICY_BLOCKED_PATTERNS = [
    # OpenAI Codex — message may arrive without an HTTP status
    "flagged for possible cybersecurity risk",
    "trusted access for cyber",
    # OpenAI moderation — chat completions / responses
    "violates our usage policies",
    "violates openai's usage policies",
    "your request was flagged by",
    # ``content_filter`` (underscore) is the OpenAI-standard error/finish
    # token surfaced verbatim by their SDKs when a request is blocked.
    # Deliberately NOT matching the space variant ("content filter") — it
    # appears in benign config descriptions and tooltip text that providers
    # echo back; the underscore form is provider-specific enough.
    "content_filter",
]

# Auth patterns (non-status-code signals)
_AUTH_PATTERNS = [
    "invalid api key",
    "invalid_api_key",
    "gateway_auth_failed",
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid token",
    "token expired",
    "token revoked",
    "access denied",
]

# Message-string patterns that indicate a provider-side timeout even when
# the exception type is generic (e.g. RuntimeError from a local shim that
# wraps a subprocess timeout).  Checked before the type-based transport
# heuristics so custom-provider "timed out" errors don't fall through to
# Provider empty-response advisories.
# Checked before context-overflow matching because the advisory text often
# mentions "max_tokens" as a possible cause, which historically sat in
# _CONTEXT_OVERFLOW_PATTERNS and sent healthy sessions into a compression
# death spiral ending in "Cannot compress further".
_EMPTY_PROVIDER_RESPONSE_PATTERNS = [
    "returned an empty response",
    "empty response despite retries",
    "provider returned an empty response",
    "model returning empty responses",
    "empty response stream",
]

# the unknown bucket and get misreported as empty responses.
_TIMEOUT_MESSAGE_PATTERNS = [
    "timed out",
    "turn timed out",
    "request timed out",
    "deadline exceeded",
    "operation timed out",
    "upstream timed out",
]

# Transport error type names
_TRANSPORT_ERROR_TYPES = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout",
    "ConnectError", "RemoteProtocolError",
    "ConnectionError", "ConnectionResetError",
    "ConnectionAbortedError", "BrokenPipeError",
    "TimeoutError", "ReadError",
    "ServerDisconnectedError",
    # SSL/TLS transport errors — transient mid-stream handshake/record
    # failures that should retry rather than surface as a stalled session.
    # ssl.SSLError subclasses OSError (caught by isinstance) but we list
    # the type names here so provider-wrapped SSL errors (e.g. when the
    # SDK re-raises without preserving the exception chain) still classify
    # as transport rather than falling through to the unknown bucket.
    "SSLError", "SSLZeroReturnError", "SSLWantReadError",
    "SSLWantWriteError", "SSLEOFError", "SSLSyscallError",
    # OpenAI SDK errors (not subclasses of Python builtins)
    "APIConnectionError",
    "APITimeoutError",
})

# Server disconnect patterns (no status code, but transport-level).
# These are the "ambiguous" patterns — a plain connection close could be
# transient transport hiccup OR server-side context overflow rejection
# (common when the API gateway disconnects instead of returning an HTTP
# error for oversized requests).  A large session + one of these patterns
# triggers the context-overflow-with-compression recovery path.
_SERVER_DISCONNECT_PATTERNS = [
    "server disconnected",
    "peer closed connection",
    "connection reset by peer",
    "connection was closed",
    "network connection lost",
    "unexpected eof",
    "incomplete chunked read",
]

# SSL certificate verification failures — deterministic, NOT transient.
#
# A failed certificate chain (TLS-inspecting corporate proxy, missing
# custom CA in the trust store, expired certificate, self-signed cert)
# fails identically on every retry. Burning the retry budget before
# surfacing the error hides the actionable fix from the user for minutes.
# Made SSL certificate
# errors fail immediately with a fix hint instead of retrying.
#
# Must be checked BEFORE _SSL_TRANSIENT_PATTERNS — "certificate verify
# failed" messages usually also contain "[SSL:" which would otherwise
# match the transient list and retry forever.
_SSL_CERT_VERIFY_PATTERNS = [
    "certificate verify failed",       # Python ssl module canonical text
    "certificate_verify_failed",       # OpenSSL error token
    "unable to get local issuer certificate",
    "self-signed certificate",
    "self signed certificate",
    "certificate has expired",
    "hostname mismatch, certificate is not valid",
    "unable to verify the first certificate",  # Node/undici phrasing (MCP bridges)
]

# SSL/TLS transient failure patterns — intentionally distinct from
# _SERVER_DISCONNECT_PATTERNS above.
#
# An SSL alert mid-stream is almost always a transport-layer hiccup
# (flaky network, mid-session TLS renegotiation failure, load balancer
# dropping the connection) — NOT a server-side context overflow signal.
# So we want the retry path but NOT the compression path; lumping these
# into _SERVER_DISCONNECT_PATTERNS would trigger unnecessary (and
# expensive) context compression on any large-session SSL hiccup.
#
# The OpenSSL library constructs error codes by prepending a format string
# to the uppercased alert reason; OpenSSL 3.x changed the separator
# (e.g. `SSLV3_ALERT_BAD_RECORD_MAC` → `SSL/TLS_ALERT_BAD_RECORD_MAC`),
# which silently stopped matching anything explicit.  Matching on the
# stable substrings (`bad record mac`, `ssl alert`, `tls alert`, etc.)
# survives future OpenSSL format churn without code changes.
_SSL_TRANSIENT_PATTERNS = [
    # Space-separated (human-readable form, Python ssl module, most SDKs)
    "bad record mac",
    "ssl alert",
    "tls alert",
    "ssl handshake failure",
    "tlsv1 alert",
    "sslv3 alert",
    # Underscore-separated (OpenSSL error code tokens, e.g.
    # `ERR_SSL_SSL/TLS_ALERT_BAD_RECORD_MAC`, `SSLV3_ALERT_BAD_RECORD_MAC`)
    "bad_record_mac",
    "ssl_alert",
    "tls_alert",
    "tls_alert_internal_error",
    # Python ssl module prefix, e.g. "[SSL: BAD_RECORD_MAC]"
    "[ssl:",
]


# ── Classification pipeline ─────────────────────────────────────────────

def classify_api_error(
    error: Exception,
    *,
    provider: str = "",
    model: str = "",
    approx_tokens: int = 0,
    context_length: int = 200000,
    num_messages: int = 0,
) -> ClassifiedError:
    """Classify an API error into a structured recovery recommendation.

    Priority-ordered pipeline:
      0. Plugin ``transform_api_error_classification`` hooks (first valid result wins)
      1. Special-case provider-specific patterns (thinking sigs, tier gates)
      2. HTTP status code + message-aware refinement
      3. Error code classification (from body)
      4. Message pattern matching (billing vs rate_limit vs context vs auth)
      5. SSL/TLS transient alert patterns → retry as timeout
      6. Server disconnect + large session → context overflow
      7. Transport error heuristics
      8. Fallback: unknown (retryable with backoff)

    Args:
        error: The exception from the API call.
        provider: Current provider name (e.g. "openai", "openai-codex").
        model: Current model slug.
        approx_tokens: Approximate token count of the current context.
        context_length: Maximum context length for the current model.

    Returns:
        ClassifiedError with reason and recovery action hints.
    """
    status_code = _extract_status_code(error)
    error_type = type(error).__name__
    # A RateLimitError may not set .status_code; force 429 so downstream
    # rate-limit handling (classifier reason, pool rotation, fallback gating)
    # fires correctly instead of misclassifying as generic.
    if status_code is None and error_type == "RateLimitError":
        status_code = 429
    body = _extract_error_body(error)
    error_code = _extract_error_code(body)

    # Build a comprehensive error message string for pattern matching.
    # str(error) alone may not include the body message (e.g. OpenAI SDK's
    # APIStatusError.__str__ returns the first arg, not the body).  Append
    # the body message so patterns like "try again" in 402 disambiguation
    # are detected even when only present in the structured body.
    _raw_msg = str(error).lower()
    _body_msg = ""
    if isinstance(body, dict):
        _err_obj = body.get("error", {})
        if isinstance(_err_obj, dict):
            _body_msg = str(_err_obj.get("message") or "").lower()
        if not _body_msg:
            _body_msg = str(body.get("message") or "").lower()
    parts = [_raw_msg]
    if _body_msg and _body_msg not in _raw_msg:
        parts.append(_body_msg)
    error_msg = " ".join(parts)
    provider_lower = (provider or "").strip().lower()
    model_lower = (model or "").strip().lower()

    def _result(reason: FailoverReason, **overrides) -> ClassifiedError:
        defaults = {
            "reason": reason,
            "status_code": status_code,
            "provider": provider,
            "model": model,
            "message": _extract_message(error, body),
        }
        defaults.update(overrides)
        return ClassifiedError(**defaults)

    # ── 0. Plugin classifiers (first valid result wins) ─────────────
    #
    # Consulted BEFORE the built-in pipeline so a provider plugin can both
    # add classifications the core patterns miss and correct ones they get
    # wrong for its provider (see the ``transform_api_error_classification`` entry in
    # pilotage_cli.plugins.VALID_HOOKS for the callback contract). Callback
    # exceptions are isolated inside invoke_hook and malformed returns are
    # dropped by the helper, so a broken plugin can never break
    # classification — the guard here only covers import/dispatch failure.
    try:
        from pilotage_cli.plugins import get_plugin_error_classification
        plugin_classification = get_plugin_error_classification(
            provider=provider,
            model=model,
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            error_message=error_msg,
            error_body=body,
            error=error,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
        )
    except Exception as exc:
        logger.debug("Plugin error classification unavailable: %s", exc)
        plugin_classification = None
    if plugin_classification is not None:
        reason = plugin_classification.pop("reason")
        logger.info(
            "API error classified by plugin hook: %s (provider=%s, status=%s)",
            reason.value, provider, status_code,
        )
        return _result(reason, **plugin_classification)

    # ── 1. Provider-specific patterns (highest priority) ────────────

    # Provider content-policy / safety-filter block. The provider has made a
    # deterministic refusal decision about THIS prompt — retrying unchanged
    # just reproduces the same refusal and burns paid attempts. Must run
    # before status-based classification so a 400 safety block isn't
    # downgraded to a generic ``format_error`` and a status-less block
    # (OpenAI Codex SDK can raise without one) isn't left in the retryable
    # ``unknown`` bucket. See.
    if any(p in error_msg for p in _CONTENT_POLICY_BLOCKED_PATTERNS):
        return _result(
            FailoverReason.content_policy_blocked,
            retryable=False,
            should_fallback=True,
        )

    # ── 2. HTTP status code classification ──────────────────────────

    if status_code is not None:
        classified = _classify_by_status(
            status_code, error_msg, error_code, body,
            provider=provider_lower, model=model_lower,
            approx_tokens=approx_tokens, context_length=context_length,
            num_messages=num_messages,
            result_fn=_result,
        )
        if classified is not None:
            return classified

    # ── 3. Error code classification ────────────────────────────────

    if error_code:
        classified = _classify_by_error_code(error_code, error_msg, _result)
        if classified is not None:
            return classified

    # ── 4. Message pattern matching (no status code) ────────────────

    classified = _classify_by_message(
        error_msg, error_type,
        approx_tokens=approx_tokens,
        context_length=context_length,
        result_fn=_result,
    )
    if classified is not None:
        return classified

    # ── 5. SSL certificate verification failures → fail fast ────────
    # A broken certificate chain (TLS-inspecting proxy, missing custom CA,
    # expired/self-signed cert) is deterministic for the host — every retry
    # reproduces the identical handshake failure. Fail immediately with
    # actionable guidance instead of burning the retry budget first.
    # Checked BEFORE the transient-SSL patterns: cert-verify messages also
    # contain "[ssl:" which would otherwise match the transient list.
    if any(p in error_msg for p in _SSL_CERT_VERIFY_PATTERNS):
        return _result(
            FailoverReason.ssl_cert_verification,
            retryable=False,
            should_fallback=False,
        )

    # ── 5b. SSL/TLS transient errors → retry as timeout (not compression) ──
    # SSL alerts mid-stream are transport hiccups, not server-side context
    # overflow signals.  Classify before the disconnect check so a large
    # session doesn't incorrectly trigger context compression when the real
    # cause is a flaky TLS handshake.  Also matches when the error is
    # wrapped in a generic exception whose message string carries the SSL
    # alert text but the type isn't ssl.SSLError (happens with some SDKs
    # that re-raise without chaining).
    if any(p in error_msg for p in _SSL_TRANSIENT_PATTERNS):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 6. Server disconnect + large session → context overflow ─────
    # Must come BEFORE generic transport error catch — a disconnect on
    # a large session is more likely context overflow than a transient
    # transport hiccup.  Without this ordering, RemoteProtocolError
    # always maps to timeout regardless of session size.

    is_disconnect = any(p in error_msg for p in _SERVER_DISCONNECT_PATTERNS)
    if is_disconnect and not status_code:
        # Reasoning-model override: a transport disconnect on a reasoning
        # model is much more likely the upstream proxy idle-killing a
        # long thinking stream than a true context overflow — even on
        # large sessions.  The default disconnect+large-session routing
        # below would otherwise send the user into the compression
        # branch (should_compress=True) and silently delete
        # conversation history on a phantom context-length error.
        # Reasoning models have multi-minute thinking phases that
        # routinely exceed the cloud gateway's idle window.  The
        # per-reasoning-model stale-timeout floor in
        # agent/reasoning_timeouts.py raises the stale-detector
        # threshold to tolerate long thinking, so a true
        # transport-layer failure here is recoverable via the retry
        # path — not via context compression.  Reclassify as timeout.
        # (Part 1 of Fixes.)
        from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
        if get_reasoning_stale_timeout_floor(model) is not None:
            return _result(FailoverReason.timeout, retryable=True)
        # Absolute token/message-count thresholds are only a proxy for smaller
        # context windows.  Large-context sessions can have hundreds of
        # messages while still being far below their actual token budget.
        is_large = approx_tokens > context_length * 0.6 or (
            context_length <= 256000 and (approx_tokens > 120000 or num_messages > 200)
        )
        if is_large:
            return _result(
                FailoverReason.context_overflow,
                retryable=True,
                should_compress=True,
            )
        return _result(FailoverReason.timeout, retryable=True)

    # ── 7b. Stale-call circuit breaker → failover immediately ──────
    # _check_stale_giveup() in agent/chat_completion_helpers.py raises a
    # RuntimeError when the provider has been unresponsive for N
    # consecutive stale attempts (default 5).  The error is NOT a transport
    # timeout — the circuit breaker fires *before* any network call to avoid
    # an indefinite stall.  Without this classification the RuntimeError
    # falls through to FailoverReason.unknown (retryable=True), which burns
    # all max_retries against the same dead provider (each retry hitting the
    # circuit breaker instantly with zero network overhead) before fallback
    # is attempted.  Classify as non-retryable + should_fallback so the
    # retry loop activates the next fallback provider on the first hit.
    if (
        error_type == "RuntimeError"
        and "consecutive stale attempts" in error_msg
        and "aborting this call" in error_msg
    ):
        return _result(
            FailoverReason.timeout,
            retryable=False,
            should_fallback=True,
        )

    # ── 8. Transport / timeout heuristics ───────────────────────────

    if error_type in _TRANSPORT_ERROR_TYPES or isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 9. Fallback: unknown ────────────────────────────────────────

    return _result(FailoverReason.unknown, retryable=True)


# ── Status code classification ──────────────────────────────────────────

def _classify_by_status(
    status_code: int,
    error_msg: str,
    error_code: str,
    body: dict,
    *,
    provider: str,
    model: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int = 0,
    result_fn,
) -> Optional[ClassifiedError]:
    """Classify based on HTTP status code with message-aware refinement."""

    if status_code == 401:
        # Not retryable on its own — credential pool rotation and
        # provider-specific refresh (Codex) run before
        # the retryability check in run_agent.py.  If those succeed, the
        # loop `continue`s.  If they fail, retryable=False ensures we
        # hit the client-error abort path (which tries fallback first).
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 403:
        # A 403 "key limit exceeded" is actually billing — providers also
        # use 403 for account-plan or credit exhaustion.
        if (
            "key limit exceeded" in error_msg
            or "spending limit" in error_msg
            or any(p in error_msg for p in _BILLING_PATTERNS)
        ):
            return result_fn(
                FailoverReason.billing,
                retryable=False,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_fallback=True,
        )

    if status_code == 402:
        return _classify_402(error_msg, result_fn)

    if status_code == 404:
        # Some endpoints surface credit depletion as a paid model becoming
        # unavailable, returned as 404 rather than 402. Treat that as
        # entitlement/billing exhaustion, not a missing model, so the retry
        # loop can show credit/top-up guidance.
        if any(p in error_msg for p in _BILLING_PATTERNS):
            return result_fn(
                FailoverReason.billing,
                retryable=False,
                should_rotate_credential=True,
                should_fallback=True,
            )
        if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
            return result_fn(
                FailoverReason.model_not_found,
                retryable=False,
                should_fallback=True,
            )
        # Generic 404 with no "model not found" signal — could be a wrong
        # endpoint path, a proxy routing glitch, or a transient backend
        # issue.  Classifying these as model_not_found
        # silently falls back to a different provider and tells the model
        # the model is missing, which is wrong and wastes a turn.  Treat
        # as unknown so the retry loop surfaces the real error instead.
        return result_fn(
            FailoverReason.unknown,
            retryable=True,
        )

    if status_code == 413:
        return result_fn(
            FailoverReason.payload_too_large,
            retryable=True,
            should_compress=True,
        )

    if status_code == 429:
        # Some providers reuse HTTP 429 for server-wide overload — same
        # status code as a true per-credential rate limit, but the credential is
        # valid and the correct recovery is "back off and retry the same key",
        # NOT "rotate the credential" (which exhausts the pool while the
        # endpoint is still busy, and does nothing for a single-key user).
        # Disambiguate on the error body so an overload 429 takes the
        # transient-overload path instead of burning the pool.
        if any(p in error_msg for p in _OVERLOADED_PATTERNS):
            return result_fn(
                FailoverReason.overloaded,
                retryable=True,
            )
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 400:
        return _classify_400(
            error_msg, error_code, body,
            provider=provider, model=model,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
            result_fn=result_fn,
        )

    if status_code in {500, 502}:
        # Some OpenAI-compatible gateways return request-validation errors
        # with a 5xx status (codex.nekos.me returns 502 for unknown/
        # unsupported parameters). These are deterministic — every retry
        # gets the identical rejection — so the generic "5xx → retryable
        # server_error" rule turns one bad request into a retry flood.
        # Detect the unambiguous request-validation signals (in either the
        # message text or the structured error code) and fail fast.
        if (
            any(p in error_msg for p in _REQUEST_VALIDATION_PATTERNS)
            or error_code.lower() in {"invalid_request_error", "unknown_parameter",
                                      "unsupported_parameter"}
        ):
            return result_fn(
                FailoverReason.format_error,
                retryable=False,
                should_fallback=True,
            )
        # Some servers report context overflow with an HTTP 500 instead of
        # the standard
        # 400/413. The request-validation guard above already ran, so any
        # remaining explicit context-overflow signal routes into the
        # compression-and-retry path (mirroring _classify_400) instead of
        # blind server_error retries that exhaust and drop the turn.
        # Empty-response advisories that mention "max_tokens" must not enter
        # that compression path.
        if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
            return result_fn(
                FailoverReason.server_error,
                retryable=True,
                should_compress=False,
            )
        if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
            return result_fn(
                FailoverReason.context_overflow,
                retryable=True,
                should_compress=True,
            )
        return result_fn(FailoverReason.server_error, retryable=True)

    if status_code in {503, 529}:
        # Same overflow-as-5xx variant (server busy / model-load OOM, or a
        # Cloudflare/Tailscale hop relabeling the status). Route explicit
        # overflow bodies into compression; otherwise treat as transient
        # overload and retry.
        if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
            return result_fn(
                FailoverReason.server_error,
                retryable=True,
                should_compress=False,
            )
        if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
            return result_fn(
                FailoverReason.context_overflow,
                retryable=True,
                should_compress=True,
            )
        return result_fn(FailoverReason.overloaded, retryable=True)

    # 408 Request Timeout — a transient timing failure the server itself flags
    # as safe to retry (RFC 9110 §15.5.9), not a malformed request. Commonly
    # emitted by reverse proxies when a long generation outruns the proxy's
    # request-read window. Route to the dedicated ``timeout`` reason (rebuild
    # client + retry) instead of falling through to the generic 4xx bucket
    # below, which would abort the turn on a retry-safe error the same way it
    # aborts a 400 Bad Request.
    if status_code == 408:
        return result_fn(FailoverReason.timeout, retryable=True)

    # Other 4xx — non-retryable
    if 400 <= status_code < 500:
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    # Other 5xx — retryable
    if 500 <= status_code < 600:
        return result_fn(FailoverReason.server_error, retryable=True)

    return None


def _classify_402(error_msg: str, result_fn) -> ClassifiedError:
    """Disambiguate 402: billing exhaustion vs transient usage limit.

    The key insight from OpenClaw: some 402s are transient rate limits
    disguised as payment errors.  "Usage limit, try again in 5 minutes"
    is NOT a billing problem — it's a periodic quota that resets.
    """
    # Check for transient usage-limit signals first
    has_usage_limit = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
    has_transient_signal = any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)

    if has_usage_limit and has_transient_signal:
        # Transient quota — treat as rate limit, not billing
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Confirmed billing exhaustion
    return result_fn(
        FailoverReason.billing,
        retryable=False,
        should_rotate_credential=True,
        should_fallback=True,
    )


def _classify_400(
    error_msg: str,
    error_code: str,
    body: dict,
    *,
    provider: str,
    model: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int = 0,
    result_fn,
) -> ClassifiedError:
    """Classify 400 Bad Request — context overflow, format error, or generic."""

    # Multimodal tool content rejected from 400.  Must be checked BEFORE
    # image_too_large because the recovery is different (strip image parts
    # from tool messages, mark the model as no-list-tool-content for the
    # rest of the session) and BEFORE context_overflow because some of the
    # patterns ("text is not set") are ambiguous in isolation but become
    # specific when combined with a 400 on a request known to contain
    # multimodal tool content.
    if any(p in error_msg for p in _MULTIMODAL_TOOL_CONTENT_PATTERNS):
        return result_fn(
            FailoverReason.multimodal_tool_content_unsupported,
            retryable=True,
        )

    # Image-too-large from 400 (per-image size checks fire this way).
    # Must be checked BEFORE context_overflow because messages can trip both
    # patterns ("exceeds" + "image") and image-shrink is a cheaper recovery.
    if any(p in error_msg for p in _IMAGE_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.image_too_large,
            retryable=True,
        )

    # Invalid encrypted reasoning replay blob (OpenAI Responses API).  Must be
    # checked BEFORE context_overflow because some surfaces emit messages that
    # contain context-like phrasing ("encrypted content … could not be
    # verified") which could otherwise trip the context_overflow heuristics.
    # ``error_msg`` is lowercased upstream — match accordingly.
    error_code_lower = (error_code or "").lower()
    if (
        error_code_lower == "invalid_encrypted_content"
        or "invalid_encrypted_content" in error_msg
        or (
            "encrypted content for item" in error_msg
            and "could not be verified" in error_msg
        )
        or "could not decrypt the provided encrypted_content" in error_msg
    ):
        return result_fn(
            FailoverReason.invalid_encrypted_content,
            retryable=True,
            should_fallback=False,
        )

    # Request-validation errors (unsupported / unknown parameter) MUST be
    # checked BEFORE context_overflow.  A GPT-5 model rejecting max_tokens
    # returns:
    #   "Unsupported parameter: 'max_tokens' is not supported with this model.
    #    Use 'max_completion_tokens' instead."
    # That string contains the literal substring "max_tokens", which historically
    # sat in _CONTEXT_OVERFLOW_PATTERNS — so without this guard the 400 is
    # misclassified as context_overflow, routed into the compression loop,
    # re-sent with the same bad parameter, and ends in "Cannot compress
    # further".  These errors are deterministic (every retry gets the identical
    # rejection), so classify as a non-retryable format_error and fall back.
    #
    # NOTE: we deliberately do NOT key off the generic ``invalid_request_error``
    # code here — OpenAI stamps that same code on genuine context-overflow 400s,
    # so matching it would mis-route real overflows away from compression. The
    # unambiguous signals are the explicit "unsupported/unknown parameter"
    # message text and the specific parameter-level error codes.
    if (
        any(p in error_msg for p in _REQUEST_VALIDATION_PATTERNS
            if p != "invalid_request_error")
        or error_code_lower in {"unknown_parameter", "unsupported_parameter"}
    ):
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    # Malformed message array (empty-content assistant stub, etc.). Must be
    # checked BEFORE context_overflow: the input can be tiny, so the generic
    # "400 + large session" heuristic would otherwise mis-route it into the
    # compression loop and thrash until "Cannot compress further" on every
    # retry (the request is unchanged, so compression cannot fix it). This is
    # a deterministic request-shape rejection — fail fast as a non-retryable
    # format_error and fall back. Checked against the message text AND the
    # structured error code, since proxies surface the signal in
    # errorCode=INVALID_REQUEST_BODY.
    if (
        any(p in error_msg for p in _INVALID_MESSAGE_BODY_PATTERNS)
        or error_code_lower == "invalid_request_body"
    ):
        logger.warning(
            "Malformed message array 400 (invalid request body) classified as "
            "format_error, NOT context overflow — failing fast + falling back "
            "instead of entering the compression loop. This usually means an "
            "empty-content assistant stub is in the transcript; num_messages=%s "
            "approx_tokens=%s. error=%.200s",
            num_messages, approx_tokens, error_msg,
        )
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    # Empty-provider-response advisories must not enter compression. They
    # often mention "max_tokens" as a possible cause and used to match the
    # bare overflow pattern, then thrash compress until "Cannot compress
    # further" on an otherwise healthy session (custom endpoints / nano-gpt).
    if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
        return result_fn(
            FailoverReason.server_error,
            retryable=True,
            should_compress=False,
        )

    # Context overflow from 400
    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    # Some providers return rate limit / billing errors as 400 instead of 429/402.
    # Check these patterns before falling through to format_error.
    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )
    if any(p in error_msg for p in _BILLING_PATTERNS):
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
            # "out of extra usage" on a 400 is ambiguous — it can also be a
        )

    # Generic 400 + large session → probable context overflow.  Providers
    # sometimes return a bare "Error" message when context is too large.
    err_body_msg = ""
    if isinstance(body, dict):
        err_obj = body.get("error", {})
        if isinstance(err_obj, dict):
            err_body_msg = str(err_obj.get("message") or "").strip().lower()
        # Responses API (and some providers) use flat body: {"message": "..."}
        if not err_body_msg:
            err_body_msg = str(body.get("message") or "").strip().lower()
    is_generic = len(err_body_msg) < 30 or err_body_msg in {"error", ""}
    # Absolute token/message-count thresholds are only a proxy for smaller
    # context windows.  Large-context sessions can have many messages while
    # still being far below their actual token budget.
    is_large = approx_tokens > context_length * 0.4 or (
        context_length <= 256000 and (approx_tokens > 80000 or num_messages > 80)
    )

    if is_generic and is_large:
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    # Non-retryable format error
    return result_fn(
        FailoverReason.format_error,
        retryable=False,
        should_fallback=True,
    )


# ── Error code classification ───────────────────────────────────────────

def _classify_by_error_code(
    error_code: str, error_msg: str, result_fn,
) -> Optional[ClassifiedError]:
    """Classify by structured error codes from the response body."""
    code_lower = error_code.lower()

    if (
        code_lower == PROVIDER_STREAM_NON_JSON_ERROR_CODE
        and "request validation failed:" in error_msg
    ):
        # Some OpenAI-compatible endpoints encode deterministic request
        # validation failures as plain-text ``event: error`` SSE data behind
        # HTTP 200.  Retrying the unchanged request cannot succeed, but a
        # configured provider fallback still may.
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    if code_lower in {"resource_exhausted", "throttled", "rate_limit_exceeded"}:
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
        )

    if code_lower in _BILLING_ERROR_CODES:
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if code_lower in {"model_not_found", "model_not_available", "invalid_model"}:
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    if code_lower in {"context_length_exceeded", "max_tokens_exceeded"}:
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    if code_lower == "invalid_encrypted_content":
        return result_fn(
            FailoverReason.invalid_encrypted_content,
            retryable=True,
            should_fallback=False,
        )

    return None


# ── Message pattern classification ──────────────────────────────────────

def _classify_by_message(
    error_msg: str,
    error_type: str,
    *,
    approx_tokens: int,
    context_length: int,
    result_fn,
) -> Optional[ClassifiedError]:
    """Classify based on error message patterns when no status code is available."""

    # Payload-too-large patterns (from message text when no status_code)
    if any(p in error_msg for p in _PAYLOAD_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.payload_too_large,
            retryable=True,
            should_compress=True,
        )

    # Multimodal tool content patterns (from message text when no status_code)
    if any(p in error_msg for p in _MULTIMODAL_TOOL_CONTENT_PATTERNS):
        return result_fn(
            FailoverReason.multimodal_tool_content_unsupported,
            retryable=True,
        )

    # Image-too-large patterns (from message text when no status_code)
    if any(p in error_msg for p in _IMAGE_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.image_too_large,
            retryable=True,
        )

    # Usage-limit patterns need the same disambiguation as 402: some providers
    # surface "usage limit" errors without an HTTP status code.  A transient
    # signal ("try again", "resets at", …) means it's a periodic quota, not
    # billing exhaustion.
    has_usage_limit = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
    if has_usage_limit:
        has_transient_signal = any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)
        if has_transient_signal:
            return result_fn(
                FailoverReason.rate_limit,
                retryable=True,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Overloaded / server-busy patterns — must come BEFORE the rate_limit and
    # billing checks so that a message-only "overloaded" (no 503/529 status,
    # e.g. some proxies) classifies as a transient
    # overload (backoff + retry) instead of falling through to `unknown` or
    # incorrectly triggering credential rotation.
    if any(p in error_msg for p in _OVERLOADED_PATTERNS):
        return result_fn(
            FailoverReason.overloaded,
            retryable=True,
        )

    # Billing patterns
    if any(p in error_msg for p in _BILLING_PATTERNS):
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Rate limit patterns
    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Empty-provider-response advisories (often mention "max_tokens") must
    # retry without compression — see the matching 400-path guard above.
    if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
        return result_fn(
            FailoverReason.server_error,
            retryable=True,
            should_compress=False,
        )

    # Context overflow patterns
    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    # Auth patterns
    # Auth errors should NOT be retried directly — the credential is invalid and
    # retrying with the same key will always fail.  Set retryable=False so the
    # caller triggers credential rotation (should_rotate_credential=True) or
    # provider fallback rather than an immediate retry loop.
    if any(p in error_msg for p in _AUTH_PATTERNS):
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Model not found patterns
    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    # Timeout message patterns — generic exception types (e.g. RuntimeError)
    # raised by local shims or custom providers that internally wrap a
    # subprocess/HTTP timeout.  Classified as transport timeout so the retry
    # loop rebuilds the client instead of treating the turn as an empty
    # model response.
    if any(p in error_msg for p in _TIMEOUT_MESSAGE_PATTERNS):
        return result_fn(FailoverReason.timeout, retryable=True)

    return None


# ── Helpers ─────────────────────────────────────────────────────────────

def _extract_status_code(error: Exception) -> Optional[int]:
    """Walk the error and its cause chain to find an HTTP status code."""
    current = error
    for _ in range(5):  # Max depth to prevent infinite loops
        code = getattr(current, "status_code", None)
        if isinstance(code, int):
            return code
        # Some SDKs use .status instead of .status_code
        code = getattr(current, "status", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
        # Walk cause chain
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return None


def _extract_error_body(error: Exception) -> dict:
    """Extract the structured error body from an SDK exception or its cause chain."""
    current = error
    for _ in range(5):  # Match _extract_status_code() traversal depth.
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            return body
        # Some errors have .response.json()
        response = getattr(current, "response", None)
        if response is not None:
            try:
                json_body = response.json()
                if isinstance(json_body, dict):
                    return json_body
            except Exception:
                pass
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return {}


def _extract_error_code(body: dict) -> str:
    """Extract an error code string from the response body."""
    if not body:
        return ""

    def _code_from_payload(payload) -> str:
        """Extract a code/type from a nested error payload dict (defensive)."""
        if not isinstance(payload, dict):
            return ""
        payload_error = payload.get("error", {})
        if isinstance(payload_error, dict):
            nested = payload_error.get("code") or payload_error.get("type") or ""
            if isinstance(nested, str) and nested.strip() and nested.strip() != "400":
                return nested.strip()
        code = payload.get("code") or payload.get("error_code") or ""
        if isinstance(code, (str, int)):
            text = str(code).strip()
            if text and text != "400":
                return text
        return ""

    error_obj = body.get("error", {})
    if isinstance(error_obj, dict):
        code = error_obj.get("code") or error_obj.get("type") or ""
        if isinstance(code, str) and code.strip() and code.strip() != "400":
            return code.strip()

        # Some providers wrap the real JSON error body as a string inside
        # error.message — peek into it for a nested code (e.g. Responses API
        # surfaces ``invalid_encrypted_content`` this way).
        message = error_obj.get("message")
        if isinstance(message, str) and message.strip().startswith("{"):
            import json
            try:
                inner = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                inner = None
            nested_code = _code_from_payload(inner)
            if nested_code:
                return nested_code

    # Top-level code
    code = body.get("code") or body.get("error_code") or body.get("errorCode") or ""
    if isinstance(code, (str, int)):
        text = str(code).strip()
        if text and text != "400":
            return text
    return ""


def _extract_message(error: Exception, body: dict) -> str:
    """Extract the most informative error message."""
    # Try structured body first
    if body:
        error_obj = body.get("error", {})
        if isinstance(error_obj, dict):
            msg = error_obj.get("message", "")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()[:500]
        msg = body.get("message", "")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()[:500]
    # Fallback to str(error)
    return str(error)[:500]
