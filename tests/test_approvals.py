"""Contract for the reduced Hermes approval queue."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from pilotage.approvals import ApprovalManager, ApprovalOutcome
from pilotage.delivery import SendResult
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
            return SendResult(True, message_id="approval-prompt")

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
            return SendResult(True, message_id="approval-prompt")

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
            return SendResult(True, message_id="approval-prompt")

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

    async def test_user_prompt_uses_the_configured_language(self):
        manager = ApprovalManager(timeout_seconds=1, language="fr")
        delivered = asyncio.Event()
        notices = []

        async def notify(text):
            notices.append(text)
            delivered.set()
            return SendResult(True, message_id="approval-prompt")

        waiting = asyncio.create_task(
            manager.request("chat", "memory", "", notify)
        )
        await asyncio.wait_for(delivered.wait(), timeout=0.2)
        self.assertIn("Approbation requise", notices[0])
        self.assertIn("Répondez /approve", notices[0])
        self.assertTrue(manager.resolve("chat", approved=False))
        await waiting

    async def test_reset_cancels_waiters_and_rejects_new_requests_until_unblocked(self):
        manager = ApprovalManager(timeout_seconds=1)
        sent = asyncio.Event()

        async def notify(_text):
            sent.set()
            return SendResult(True, message_id="approval-prompt")

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
        sent.clear()
        fresh = asyncio.create_task(
            manager.request("chat", "memory", "write", notify)
        )
        await asyncio.wait_for(sent.wait(), timeout=0.2)
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

    async def test_retryable_prompt_delivery_retries_the_exact_prompt(self):
        manager = ApprovalManager(timeout_seconds=1)
        delivered = asyncio.Event()
        prompts = []

        async def notify(text):
            prompts.append(text)
            if len(prompts) == 1:
                return SendResult(
                    False,
                    "flood control",
                    retryable=True,
                    retry_after=0,
                )
            delivered.set()
            return SendResult(True, message_id="approval-prompt")

        with (
            mock.patch("pilotage.delivery.random.uniform", return_value=0),
            mock.patch(
                "pilotage.delivery.asyncio.sleep", new=mock.AsyncMock()
            ) as sleep,
        ):
            waiting = asyncio.create_task(
                manager.request("chat", "cron", "create report", notify)
            )
            await asyncio.wait_for(delivered.wait(), timeout=0.2)
            self.assertTrue(manager.resolve("chat", approved=True))
            outcome = await waiting

        self.assertTrue(outcome.approved)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0], prompts[1])
        sleep.assert_awaited_once_with(0.0)

    async def test_prompt_retry_never_outlives_the_approval_deadline(self):
        manager = ApprovalManager(timeout_seconds=0.02)
        attempts = 0

        async def notify(_text):
            nonlocal attempts
            attempts += 1
            return SendResult(
                False,
                "flood control",
                retryable=True,
                retry_after=1,
            )

        with (
            mock.patch("pilotage.delivery.random.uniform", return_value=0),
            mock.patch(
                "pilotage.delivery.asyncio.sleep", new=mock.AsyncMock()
            ) as sleep,
        ):
            outcome = await manager.request(
                "chat", "skills", "install reviewed skill", notify
            )

        self.assertEqual(outcome.status, "unavailable")
        self.assertEqual(attempts, 1)
        sleep.assert_not_awaited()
        self.assertFalse(manager.resolve("chat", approved=True))


class ToolContextApprovalTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_approval_values_never_invoke_client_callbacks(self):
        request = mock.AsyncMock(side_effect=AssertionError("No client approval"))
        for legacy in (True, False):
            for category in ("memory", "skills", "cron"):
                with self.subTest(legacy=legacy, category=category):
                    context = ToolContext(
                        "chat",
                        SimpleNamespace(settings=Settings({"approvals": {category: legacy}})),
                        approval_request=request,
                        persistence_writes_allowed=True,
                    )
                    self.assertTrue((await context.authorize(category, "technical proposal")).approved)
        request.assert_not_awaited()

    async def test_disabled_capability_never_asks_for_confirmation(self):
        request = mock.AsyncMock(side_effect=AssertionError("No client approval"))
        for category in ("memory", "skills", "cron"):
            for legacy in (True, False):
                with self.subTest(category=category, legacy=legacy):
                    context = ToolContext(
                        "chat",
                        SimpleNamespace(settings=Settings({
                            "tools": {"disabled": [category]},
                            "approvals": {category: legacy},
                        })),
                        approval_request=request,
                        persistence_writes_allowed=True,
                    )
                    self.assertFalse((await context.authorize(category, "proposal")).approved)
        request.assert_not_awaited()

    async def test_disabled_capability_message_uses_the_profile_language(self):
        context = ToolContext(
            "chat", SimpleNamespace(
                language="fr", settings=Settings({"tools": {"disabled": ["memory"]}})
            ), persistence_writes_allowed=True,
        )
        outcome = await context.authorize("memory", "technical proposal")
        self.assertFalse(outcome.approved)
        self.assertEqual(outcome.message, "Cette fonctionnalité n’est pas disponible.")

    async def test_runtime_limits_and_cron_switch_cannot_be_widened(self):
        context = ToolContext(
            "chat", SimpleNamespace(settings=Settings({"cron": {"enabled": False}})),
            allowed_tool_groups=frozenset({"file", "memory", "cron"}),
            persistence_writes_allowed=True,
        )
        self.assertFalse((await context.authorize("skills", "proposal")).approved)
        self.assertFalse((await context.authorize("cron", "proposal")).approved)
        context.persistence_writes_allowed = False
        self.assertFalse((await context.authorize("memory", "proposal")).approved)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
