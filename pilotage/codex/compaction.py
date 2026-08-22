"""Native Responses compaction for the fixed ChatGPT Codex route.

This is the narrow part of Hermes' ``agent/native_compaction.py`` that this
runtime needs. The backend is fixed by ``pilotage.codex.client``; only the
model-family gate, request directive, checkpoint replay, and retained-user
budget belong here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DEFAULT_COMPACT_THRESHOLD = 200_000
RETAINED_USER_MESSAGE_TOKEN_BUDGET = 64_000

_ELIGIBLE_MODEL_MARKER = "gpt-5.6"


def is_native_compaction_model(model: Optional[str]) -> bool:
    """Return whether Hermes has verified native compaction for this model."""
    return _ELIGIBLE_MODEL_MARKER in (model or "").lower()


def context_management(
    *, model: str, enabled: bool, compact_threshold: int
) -> Optional[List[Dict[str, Any]]]:
    """Build the Responses directive, or omit it byte-for-byte when gated."""
    if not enabled or not is_native_compaction_model(model):
        return None
    threshold = int(compact_threshold)
    if threshold <= 0:
        threshold = DEFAULT_COMPACT_THRESHOLD
    return [{"type": "compaction", "compact_threshold": threshold}]


def has_compaction_checkpoint(items: Any) -> bool:
    return any(
        isinstance(item, dict)
        and item.get("type") == "compaction"
        and isinstance(item.get("encrypted_content"), str)
        and bool(item["encrypted_content"])
        for item in (items if isinstance(items, list) else ())
    )


def persistent_compaction_items(items: Any) -> List[Dict[str, str]]:
    """Keep only the opaque checkpoints that must survive a process restart."""
    return [
        {"type": "compaction", "encrypted_content": item["encrypted_content"]}
        for item in (items if isinstance(items, list) else ())
        if isinstance(item, dict)
        and item.get("type") == "compaction"
        and isinstance(item.get("encrypted_content"), str)
        and bool(item["encrypted_content"])
    ]


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _user_item_text(item: Dict[str, Any]) -> Optional[str]:
    content = item.get("content")
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "input_text"
        )
        return text if text.strip() or content else None
    return None


def prune_pre_checkpoint_items(
    items: List[Dict[str, Any]],
    retained_user_token_budget: int = RETAINED_USER_MESSAGE_TOKEN_BUDGET,
) -> List[Dict[str, Any]]:
    """Replay the newest checkpoint, retained user asks, then its live tail.

    The Responses backend renders nothing placed before a replayed compaction
    item. This is Hermes' wire ordering and retention rule.
    """
    last_checkpoint: Optional[int] = None
    for index, item in enumerate(items):
        if isinstance(item, dict) and item.get("type") == "compaction":
            last_checkpoint = index
    if last_checkpoint is None:
        return items

    first_checkpoint = last_checkpoint
    while (
        first_checkpoint > 0
        and isinstance(items[first_checkpoint - 1], dict)
        and items[first_checkpoint - 1].get("type") == "compaction"
    ):
        first_checkpoint -= 1

    pre = items[:first_checkpoint]
    checkpoint_run = items[first_checkpoint : last_checkpoint + 1]
    post = items[last_checkpoint + 1 :]

    retained_reversed: List[Dict[str, Any]] = []
    remaining = max(0, int(retained_user_token_budget))
    for item in reversed(pre):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        if "type" in item and item.get("type") != "message":
            continue
        if remaining <= 0:
            break
        text = _user_item_text(item)
        if text is None:
            continue
        cost = _approx_tokens(text)
        if cost <= remaining:
            retained_reversed.append(item)
            remaining -= cost
        elif isinstance(item.get("content"), str):
            truncated = dict(item)
            truncated["content"] = item["content"][: remaining * 4]
            if truncated["content"].strip():
                retained_reversed.append(truncated)
            remaining = 0

    return checkpoint_run + list(reversed(retained_reversed)) + post


def prepare_input_items(
    items: List[Dict[str, Any]], *, native_compaction_active: bool
) -> List[Dict[str, Any]]:
    """Gate checkpoint replay and wire pruning from the same request decision."""
    if not native_compaction_active:
        return [
            item
            for item in items
            if not (isinstance(item, dict) and item.get("type") == "compaction")
        ]
    return prune_pre_checkpoint_items(items)
