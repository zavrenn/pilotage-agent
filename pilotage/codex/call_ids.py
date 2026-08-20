"""Identifiers for tool calls on the Responses wire.

Copied from the Hermes agent (``agent/message_sanitization.py`` and
``agent/codex_responses_adapter.py``), where the rules below were learned the
hard way. The invariant is worth repeating: these ids are part of the request
prefix, so they must be derived from content and never from a random source. A
uuid4 here would give every request a unique prefix and throw away the prompt
cache on every single turn.
"""

from __future__ import annotations

import hashlib

# The Responses API rejects an item id longer than this.
MAX_ITEM_ID_LENGTH = 64


def deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
    """A call id derived from the call itself, for when the API omits one.

    Deterministic IDs prevent cache invalidation — random UUIDs would make
    every API call's prefix unique, breaking OpenAI's prompt cache.
    """
    seed = f"{fn_name}:{arguments}:{index}"
    digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"call_{digest}"


def clamp_call_id(call_id: str) -> str:
    """Shorten an over-long call id without losing which call it belongs to."""
    if len(call_id) <= MAX_ITEM_ID_LENGTH:
        return call_id
    digest = hashlib.sha256(call_id.encode("utf-8", errors="replace")).hexdigest()[:32]
    return f"call_{digest}"
