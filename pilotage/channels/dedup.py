"""TTL message deduplication.

Copied from Hermes' ``gateway/platforms/helpers.py`` (MessageDeduplicator),
which every one of its adapters relies on. The bridge can hand the same
message id back more than once — a ``messages.upsert`` of type ``append``
replays recent history after a reconnect, and a poll that overlaps a restart
sees the queue twice. Answering the same question twice is the visible cost.
"""

from __future__ import annotations

import time
from typing import Dict


class MessageDeduplicator:
    """TTL-based message deduplication cache."""

    def __init__(self, max_size: int = 2000, ttl_seconds: float = 300):
        self._seen: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def is_duplicate(self, msg_id: str) -> bool:
        """Return True if *msg_id* was already seen within the TTL window."""
        if not msg_id:
            return False
        now = time.time()
        if msg_id in self._seen:
            if now - self._seen[msg_id] < self._ttl:
                return True
            # Entry has expired — remove it and treat as new
            del self._seen[msg_id]
        self._seen[msg_id] = now
        if len(self._seen) > self._max_size:
            cutoff = now - self._ttl
            self._seen = {k: v for k, v in self._seen.items() if v > cutoff}
            if len(self._seen) > self._max_size:
                # TTL pruning alone does not cap the cache when every entry is
                # still fresh. Keep the newest entries so the helper's
                # max_size bound is enforced under sustained traffic.
                newest = sorted(
                    self._seen.items(),
                    key=lambda item: item[1],
                )[-self._max_size:]
                self._seen = dict(newest)
        return False

    def contains(self, msg_id: str) -> bool:
        """Return whether *msg_id* is live in the cache without inserting it."""
        if not msg_id:
            return False
        seen_at = self._seen.get(msg_id)
        if seen_at is None:
            return False
        if time.time() - seen_at < self._ttl:
            return True
        del self._seen[msg_id]
        return False

    def discard(self, msg_id: str) -> None:
        """Release a claimed message ID after cancelled/failed handoff."""
        self._seen.pop(msg_id, None)

    def clear(self) -> None:
        """Clear all tracked messages."""
        self._seen.clear()
