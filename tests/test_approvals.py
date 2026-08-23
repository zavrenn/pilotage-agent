"""Contract for the reduced Hermes approval queue."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from pilotage.approvals import ApprovalManager, ApprovalOutcome
from pilotage.settings import Settings
from pilotage.tools import ToolContext


class ApprovalManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_blocks_until_the_same_session_approves(self):
        manager = ApprovalManager(timeout_seconds=1)
        sent = asyncio.Event()
        notices = []

        async def notify(text):
            notices.append(text)
            sent.set()

        waiting = asyncio.create_task(
            manager.request("chat-a", "memory", "Add: concise replies", notify)
        )
        await asyncio.wait_for(sent.wait(), timeout=0.2)

        self.assertTrue(manager.has_pending("chat-a"))
        self.assertFalse(manager.resolve("chat-b", approved=True))
        self.assertTrue(manager.resolve("chat-a", approved=True))
        outcome = await waiting

        self.assertTrue(outcome.approved)
        self.assertIn("Add: concise replies", notices[0])
        self.assertIn("/approve", notices[0])
        self.assertFalse(manager.has_pending("chat-a"))

    async def test_requests_are_resolved_fifo_and_denial_reason_is_preserved(self):
        manager = ApprovalManager(timeout_seconds=1)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        first_delivered = asyncio.Event()
        second_delivered = asyncio.Event()

        async def notify(text):
            if "first" in text:
                first_started.set()
                await release_first.wait()
                first_delivered.set()
            else:
                second_delivered.set()

        first = asyncio.create_task(
            manager.request("chat", "memory", "first", notify)
        )
        second = asyncio.create_task(
            manager.request("chat", "cron", "second", notify)
        )
        await asyncio.wait_for(first_started.wait(), timeout=0.2)

        self.assertEqual(manager.pending_count("chat"), 1)
        self.assertFalse(second_delivered.is_set())
        self.assertFalse(manager.resolve("chat", approved=True))

        release_first.set()
        await asyncio.wait_for(first_delivered.wait(), timeout=0.2)

        self.assertTrue(manager.resolve("chat", approved=False, reason="Not now"))
        denied = await first
        self.assertFalse(denied.approved)
        self.assertEqual(denied.message, "Not now")

        await asyncio.wait_for(second_delivered.wait(), timeout=0.2)
        self.assertEqual(manager.pending_count("chat"), 1)

        self.assertTrue(manager.resolve("chat", approved=True))
        self.assertTrue((await second).approved)

    async def test_timeout_and_missing_surface_fail_closed(self):
        manager = ApprovalManager(timeout_seconds=0.01)

        async def notify(_text):
            return None

        timed_out = await manager.request("chat", "skills", "write", notify)
        unavailable = await manager.request("chat", "skills", "write", None)

        self.assertEqual(timed_out.status, "timed out")
        self.assertEqual(unavailable.status, "unavailable")
        self.assertFalse(manager.has_pending("chat"))

    async def test_failed_notification_makes_the_request_unavailable(self):
        manager = ApprovalManager(timeout_seconds=1)

        async def notify(_text):
            raise RuntimeError("delivery rejected")

        outcome = await manager.request("chat", "skills", "write", notify)

        self.assertEqual(outcome.status, "unavailable")
        self.assertFalse(manager.has_pending("chat"))
        self.assertFalse(manager.resolve("chat", approved=True))

    async def test_reset_cancels_waiters_and_rejects_new_requests_until_unblocked(self):
        manager = ApprovalManager(timeout_seconds=1)
        sent = asyncio.Event()

        async def notify(_text):
            sent.set()

        waiting = asyncio.create_task(
            manager.request("chat", "memory", "write", notify)
        )
        await asyncio.wait_for(sent.wait(), timeout=0.2)
        queued = asyncio.create_task(
            manager.request("chat", "cron", "queued", notify)
        )
        await asyncio.sleep(0)
        manager.block("chat")

        self.assertEqual((await waiting).status, "cancelled")
        self.assertEqual((await queued).status, "cancelled")
        blocked = await manager.request("chat", "cron", "change", notify)
        self.assertEqual(blocked.status, "cancelled")

        manager.unblock("chat")
        fresh = asyncio.create_task(
            manager.request("chat", "memory", "write", notify)
        )
        await asyncio.sleep(0)
        self.assertTrue(manager.resolve("chat", approved=True))
        self.assertTrue((await fresh).approved)

    async def test_cancelling_during_notification_drops_the_pending_entry(self):
        manager = ApprovalManager(timeout_seconds=1)
        notifying = asyncio.Event()

        async def notify(_text):
            notifying.set()
            await asyncio.Event().wait()

        waiting = asyncio.create_task(
            manager.request("chat", "memory", "write", notify)
        )
        await asyncio.wait_for(notifying.wait(), timeout=0.2)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertFalse(manager.has_pending("chat"))


class ToolContextApprovalTests(unittest.IsolatedAsyncioTestCase):
    async def test_category_switch_can_bypass_or_require_the_live_callback(self):
        calls = []

        async def request(category, summary):
            calls.append((category, summary))
            return ApprovalOutcome(True, "approved")

        disabled = ToolContext(
            "chat",
            SimpleNamespace(settings=Settings({"approvals": {"memory": False}})),
            approval_request=request,
        )
        enabled = ToolContext(
            "chat",
            SimpleNamespace(settings=Settings({"approvals": {"memory": True}})),
            approval_request=request,
        )

        self.assertTrue((await disabled.authorize("memory", "first")).approved)
        self.assertEqual(calls, [])
        self.assertTrue((await enabled.authorize("memory", "second")).approved)
        self.assertEqual(calls, [("memory", "second")])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
