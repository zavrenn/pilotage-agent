"""Small, Hermes-derived approval queue for persistent agent changes.

Hermes blocks an active tool call, sends the proposed change to the messaging
surface, and lets ``/approve`` or ``/deny`` resolve that exact session's oldest
pending request.  Pilotage keeps that mechanism and drops the framework-wide
policy engine: only the three production write classes use it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .delivery import send_with_retry
from .i18n import DEFAULT_LANGUAGE, t

logger = logging.getLogger(__name__)

APPROVAL_CATEGORIES = frozenset({"memory", "skills", "cron"})
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300.0

Notify = Callable[[str], Awaitable[Any]]


@dataclass(frozen=True)
class ApprovalOutcome:
    """The result returned to a tool waiting at an approval gate."""

    approved: bool
    status: str
    message: str = ""


@dataclass
class _PendingApproval:
    category: str
    summary: str
    future: "asyncio.Future[ApprovalOutcome]"
    notified: bool = False


def approval_required(config: Any, category: str) -> bool:
    """Read one category's per-profile switch, defaulting safely to on."""

    if category not in APPROVAL_CATEGORIES:
        raise ValueError(f"Unknown approval category: {category}")
    attribute = {
        "memory": "approval_memory",
        "skills": "approval_skills",
        "cron": "approval_cron",
    }[category]
    configured = getattr(config, attribute, None)
    if isinstance(configured, bool):
        return configured
    settings = getattr(config, "settings", None)
    if settings is not None:
        return settings.flag(f"approvals.{category}", True)
    return True


def approval_error(outcome: ApprovalOutcome) -> str:
    """A concise tool-facing refusal that discourages approval-loop retries."""

    detail = outcome.message.strip() or "The change was not approved."
    return (
        f"Approval {outcome.status}: {detail} Nothing was changed. "
        "Do not retry this change unless the user asks for it again."
    )


class ApprovalManager:
    """Per-agent FIFO approval queues, one isolated queue per conversation."""

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        language: str = DEFAULT_LANGUAGE,
    ):
        self._timeout_seconds = float(timeout_seconds)
        self._language = str(language or DEFAULT_LANGUAGE)
        self._pending: Dict[str, List[_PendingApproval]] = {}
        self._blocked_sessions: Dict[str, int] = {}
        self._request_locks: Dict[str, asyncio.Lock] = {}

    def has_pending(self, session_id: str) -> bool:
        return bool(self._pending.get(session_id))

    def pending_count(self, session_id: str) -> int:
        return len(self._pending.get(session_id, ()))

    async def request(
        self,
        session_id: str,
        category: str,
        summary: str,
        notify: Optional[Notify],
    ) -> ApprovalOutcome:
        """Notify the user and wait for a same-session command response."""

        if category not in APPROVAL_CATEGORIES:
            raise ValueError(f"Unknown approval category: {category}")
        if session_id in self._blocked_sessions:
            return ApprovalOutcome(
                False,
                "cancelled",
                "The conversation is being reset.",
            )
        if not session_id or notify is None:
            return ApprovalOutcome(
                False,
                "unavailable",
                "This turn has no interactive messaging channel for approval.",
            )

        request_lock = self._request_locks.setdefault(session_id, asyncio.Lock())
        async with request_lock:
            if session_id in self._blocked_sessions:
                return ApprovalOutcome(
                    False,
                    "cancelled",
                    "The conversation is being reset.",
                )

            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._timeout_seconds
            entry = _PendingApproval(
                category=category,
                summary=summary.strip(),
                future=loop.create_future(),
            )
            self._pending.setdefault(session_id, []).append(entry)

            prompt = (
                f"{t('approval.required', self._language, category=category)}\n\n"
                f"{entry.summary or t('approval.default_summary', self._language)}\n\n"
                f"{t('approval.instructions', self._language)}"
            )
            try:
                delivery = await send_with_retry(
                    lambda: notify(prompt),
                    deadline=deadline,
                )
                if not delivery.success:
                    if delivery.error == "delivery deadline expired":
                        raise asyncio.TimeoutError
                    raise RuntimeError(
                        delivery.error or "approval prompt delivery rejected"
                    )
                entry.notified = True
            except asyncio.CancelledError:
                self._remove(session_id, entry)
                if not entry.future.done():
                    entry.future.cancel()
                raise
            except asyncio.TimeoutError:
                self._remove(session_id, entry)
                if not entry.future.done():
                    entry.future.cancel()
                return ApprovalOutcome(
                    False,
                    "timed out",
                    "The approval request could not be delivered before the timeout.",
                )
            except Exception as exc:  # noqa: BLE001 - failure must close the gate
                logger.warning(
                    "Could not send approval request: %s", exc
                )
                self._remove(session_id, entry)
                if not entry.future.done():
                    entry.future.cancel()
                return ApprovalOutcome(
                    False,
                    "unavailable",
                    "The approval request could not be delivered.",
                )

            try:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return ApprovalOutcome(
                        False,
                        "timed out",
                        "The approval request deadline expired after delivery.",
                    )
                return await asyncio.wait_for(
                    asyncio.shield(entry.future), remaining
                )
            except asyncio.CancelledError:
                if not entry.future.done():
                    entry.future.cancel()
                raise
            except asyncio.TimeoutError:
                if not entry.future.done():
                    entry.future.cancel()
                return ApprovalOutcome(
                    False,
                    "timed out",
                    "No approval response arrived before the timeout.",
                )
            finally:
                self._remove(session_id, entry)

    def resolve(
        self,
        session_id: str,
        *,
        approved: bool,
        reason: str = "",
    ) -> bool:
        """Resolve the oldest request for a conversation, matching Hermes FIFO."""

        queue = self._pending.get(session_id)
        while queue:
            entry = queue[0]
            if entry.future.done():
                queue.pop(0)
                continue
            if not entry.notified:
                return False
            queue.pop(0)
            if not queue:
                self._pending.pop(session_id, None)
            break
        else:
            self._pending.pop(session_id, None)
            return False
        if approved:
            outcome = ApprovalOutcome(True, "approved")
        else:
            outcome = ApprovalOutcome(
                False,
                "denied",
                reason.strip()[:280] or "The user denied this change.",
            )
        entry.future.set_result(outcome)
        return True

    def clear(
        self,
        session_id: str,
        reason: str = "The conversation was reset.",
    ) -> int:
        """Deny every waiter at a real conversation boundary."""

        queue = self._pending.pop(session_id, [])
        for entry in queue:
            if not entry.future.done():
                entry.future.set_result(ApprovalOutcome(False, "cancelled", reason))
        return len(queue)

    def block(self, session_id: str) -> None:
        """Cancel current waits and reject new ones while a reset waits for its lock."""

        self._blocked_sessions[session_id] = (
            self._blocked_sessions.get(session_id, 0) + 1
        )
        self.clear(session_id)

    def unblock(self, session_id: str) -> None:
        remaining = self._blocked_sessions.get(session_id, 0) - 1
        if remaining > 0:
            self._blocked_sessions[session_id] = remaining
        else:
            self._blocked_sessions.pop(session_id, None)

    def _remove(self, session_id: str, entry: _PendingApproval) -> None:
        queue = self._pending.get(session_id)
        if not queue:
            return
        try:
            queue.remove(entry)
        except ValueError:
            return
        if not queue:
            self._pending.pop(session_id, None)


__all__ = [
    "APPROVAL_CATEGORIES",
    "DEFAULT_APPROVAL_TIMEOUT_SECONDS",
    "ApprovalManager",
    "ApprovalOutcome",
    "approval_error",
    "approval_required",
]
