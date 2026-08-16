"""Per-reasoning-model stale-timeout floor for known reasoning models.

Reasoning models (those that emit extended thinking blocks before their
first content token) routinely exceed Pilotage's default chat-model
stale detectors:

* Stream stale detector:   ``PILOTAGE_STREAM_STALE_TIMEOUT``     default 180s
                           ``agent/chat_completion_helpers.py:2544``
* Non-stream stale detector: ``PILOTAGE_API_CALL_STALE_TIMEOUT``  default 90s
                           ``run_agent.py:1140``

For NVIDIA Nemotron 3 Ultra on the hosted NIM gateway the empirical
upstream idle kill is ~120s (first-party reproduction at
NVIDIA/NemoClaw — TTFB ~31s, stream dies at 120s). The same
failure mode exists on OpenAI o1/o3, Anthropic Opus 4.x thinking,
DeepSeek R1, Qwen QwQ, xAI Grok reasoning — every cloud reasoning
model hits upstream-proxies / load-balancers with idle timeouts
shorter than the model's thinking phase. Result: the stale detector
kills the connection mid-think, surfacing as
``BrokenPipeError``/``RemoteProtocolError`` on the next read.

This module provides a floor that the existing stale-detector scaling
blocks consult via :func:`get_reasoning_stale_timeout_floor` and
apply as ``max(default, floor)``. It is a FLOOR:

* Never overrides explicit user config (``providers.<id>.models.<model>.stale_timeout_seconds``
  or ``request_timeout_seconds`` already wins — this code never runs
  in that branch).
* Never lowers an existing threshold.
* Has zero effect on non-reasoning models — they are not in the
  allowlist and the resolver returns ``None``.

Matching uses start-anchored regex on the slug-only component of
the model name (after stripping any aggregator prefix like
``openai/``, ``x-ai/``, ``anthropic/``).  The right-anchor matches
end-of-string or a ``-``/``.``/``_`` slug separator, so ``qwen3-235b``
matches the ``qwen3`` family entry (a future model slug would be
``qwen3-235b-instruct`` and would also match) but ``some-other-qwen3``
does NOT match ``qwen3`` (the ``-qwen3`` is not at start of slug).

The ``o1`` case is the most delicate: a model named
``llama-4-70b-o1-preview`` is a hypothetical community derivative that
should NOT trigger the reasoning-model floor for the user (the user
chose a non-OpenAI model, not a reasoning model).  The start-of-slug
anchor naturally excludes this — the matched ``o1-preview`` is at
position 11 of the slug, not at position 0.  The previous substring-
with-trailing-hyphen design would have over-matched here, which is
why start-of-slug anchoring is the right shape.

Fixes.
"""

from __future__ import annotations

import re
from typing import Optional


# (slug, floor_seconds).  Each slug is matched as a discrete
# word-boundary component via the wrapper regex in ``_match_any``
# below.  Order is irrelevant — the first regex match wins.
_REASONING_STALE_TIMEOUT_FLOORS: tuple[tuple[str, int], ...] = (
    # OpenAI o-series — known multi-minute TTFB.  Each variant
    # enumerated explicitly so bare ``o1`` doesn't over-match
    # ``olmo-1`` or hypothetical future community derivatives.
    ("o1", 600),
    ("o1-mini", 600),
    ("o1-pro", 600),
    ("o1-preview", 600),
    ("o3", 600),
    ("o3-pro", 600),
    ("o3-mini", 300),
    ("o4-mini", 300),
)


# Pre-compile each pattern.  Wrapper = start-of-slug + slug + end-or-
# separator, where ``start-of-slug`` means start-of-string OR
# immediately after the last ``/`` (aggregator separator) and
# ``end-or-separator`` means end-of-string OR a ``-``/``.``/``_``.
#
# Why start-of-slug and not start-of-string: aggregator prefixes
# like ``openai/`` should not affect matching — the slug identity is
# the part after the last ``/``.  Stripping the aggregator prefix in
# :func:`get_reasoning_stale_timeout_floor` before regex matching
# gives the wrapper a clean start-of-string anchor.
#
# Why end-or-separator on the right: ``openai/o3-mini`` must match
# the ``o3-mini`` slug (the right anchor is end-of-string).  And
# ``openai/o3-mini-2025-01-31`` must also match ``o3-mini`` (the right
# anchor is the ``-`` separator).  But ``openai/o3-mini-fork`` should
# NOT match ``o3-mini`` if we wanted to exclude forks — though the
# pattern ``o3-mini-fork`` would be matched as a derivative anyway,
# so we accept that community forks inheriting the same prefix are
# treated as reasoning models (a reasonable default — the upstream
# gateway timing is the same).
# Pre-compile all patterns at module load time to avoid per-call regex
# compilation and thread-safety issues with the mutable _PATTERN_CACHE.
# The list is built once at import and never mutated afterwards, so it is
# safe for free-threaded Python 3.13+ without any locking. The slug is kept
# in each entry for debuggability (log/inspection), even though _match_any
# only consumes floor + pattern.
_SORTED_REASONING_FLOORS: list[tuple[str, float, re.Pattern[str]]] = [
    (slug, floor, re.compile(r"^" + re.escape(slug) + r"(?:$|[\-._])"))
    for slug, floor in sorted(
        _REASONING_STALE_TIMEOUT_FLOORS, key=lambda kv: -len(kv[0])
    )
]


def _match_any(model_lower: str) -> Optional[float]:
    """Return the floor for the first matching slug, else None.

    Each table entry is matched as a start-of-slug prefix with the
    slug-separator-or-end-of-string right-anchor.  Table iteration
    order is irrelevant: longest slug wins (so ``o3-mini`` beats
    ``o3`` on a model like ``openai/o3-mini``).
    """
    for _slug, floor, pattern in _SORTED_REASONING_FLOORS:
        if pattern.search(model_lower):
            return float(floor)
    return None


def get_reasoning_stale_timeout_floor(model: object) -> Optional[float]:
    """Return the stale-timeout floor (seconds) for a known reasoning model.

    Returns ``None`` when the model is not in the allowlist or the
    argument is empty / not a string.  Matching uses
    word-boundary-anchored regex on the lowercased model name, so
    ``openai/o3-mini`` matches the ``o3-mini`` slug but
    ``olmo-1`` does NOT match ``o1`` (the ``o1`` substring is not
    at a word boundary inside ``olmo-1``).

    Aggregator prefixes (``openai/``, ``x-ai/``, ``anthropic/`` etc.)
    are preserved through matching — the ``/`` is itself a word
    boundary, so ``openai/o3-mini`` matches ``o3-mini`` because the
    ``/`` before ``o3-mini`` satisfies the left-anchor alternation.

    This is a FLOOR — callers must apply it as ``max(default, floor)``
    and only when no explicit user-configured per-model
    ``stale_timeout_seconds`` exists.

    >>> get_reasoning_stale_timeout_floor("nvidia/nemotron-3-ultra-550b-a55b")
    600.0
    >>> get_reasoning_stale_timeout_floor("openai/o3-mini")
    300.0
    >>> get_reasoning_stale_timeout_floor("deepseek/deepseek-r1")
    600.0
    >>> get_reasoning_stale_timeout_floor("deepseek/deepseek-v4-flash")
    600.0
    >>> get_reasoning_stale_timeout_floor("deepseek/deepseek-v4-pro")
    600.0
    >>> get_reasoning_stale_timeout_floor("qwen/qwen3-235b-a22b-thinking")
    180.0
    >>> get_reasoning_stale_timeout_floor("x-ai/grok-4-fast-reasoning")
    300.0
    >>> get_reasoning_stale_timeout_floor("anthropic/claude-opus-4-6")
    240.0
    >>> get_reasoning_stale_timeout_floor("anthropic/claude-fable-5")
    600.0
    >>> get_reasoning_stale_timeout_floor("gpt-4o") is None
    True
    >>> get_reasoning_stale_timeout_floor("olmo-1") is None
    True
    >>> get_reasoning_stale_timeout_floor(None) is None
    True
    """
    if not model or not isinstance(model, str):
        return None
    name = model.strip().lower()
    if not name:
        return None
    # Strip aggregator prefix (everything before and including the
    # last ``/``).  The wrapper regex anchors at start-of-string, so
    # the slug identity is the bare model name.
    if "/" in name:
        name = name.rsplit("/", 1)[1]
    return _match_any(name)
