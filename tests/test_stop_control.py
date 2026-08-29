"""Exact-session stop behavior across live tasks and durable turn state."""

from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from pilotage.agent import Agent, StopStatus, TurnStopped
from pilotage.codex import stream as codex_stream
from pilotage.config import Config
from pilotage.delivery import SendResult
from pilotage.history import ConversationError, ConversationStore
from pilotage.i18n import t


class StopStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = ConversationStore(Path(temporary.name) / "conversations.db")

    def test_tool_requested_stop_becomes_an_unknown_fence(self) -> None:
        session = self.store.begin_turn("chat", "change the external system")
        self.store.checkpoint_turn(
            "chat",
            "change the external system",
            [
                {
                    "type": "function_call",
                    "name": "external_write",
                    "call_id": "call_1",
                    "arguments": "{}",
                }
            ],
            phase="tool_requested",
            iteration=1,
        )

        checkpoint = self.store.request_stop(
            "chat",
            "Stopped.",
            expected_session=session,
        )

        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint.state, "unknown")
        self.assertEqual(checkpoint.previous_phase, "tool_requested")
        self.assertEqual(self.store.list_active_turns()[0].phase, "unknown")
        with self.assertRaisesRegex(ConversationError, "previous turn"):
            self.store.begin_turn("chat", "do something else")
        with self.assertRaisesRegex(ValueError, "stopped checkpoint"):
            self.store.complete_stopped_turn(checkpoint)

    def test_tool_completed_stop_records_the_stronger_marker(self) -> None:
        user_text = "write the record"
        session = self.store.begin_turn("chat", user_text)
        call = {
            "type": "function_call",
            "name": "external_write",
            "call_id": "call_1",
            "arguments": "{}",
        }
        self.store.checkpoint_turn(
            "chat",
            user_text,
            [call],
            phase="tool_requested",
            iteration=1,
        )
        self.store.checkpoint_turn(
            "chat",
            user_text,
            [
                call,
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "written",
                },
            ],
            phase="tool_completed",
            iteration=1,
        )

        checkpoint = self.store.request_stop(
            "chat",
            "Stopped before acting.",
            stopped_after_actions_text="Stopped after completed actions.",
            expected_session=session,
        )

        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint.state, "stopped")
        self.assertEqual(checkpoint.previous_phase, "tool_completed")
        self.assertEqual(
            self.store.complete_stopped_turn(checkpoint),
            (user_text, "Stopped after completed actions."),
        )
        self.assertEqual(
            self.store.load("chat", 2),
            [
                ("user", user_text),
                ("assistant", "Stopped after completed actions."),
            ],
        )


class StopAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = ConversationStore(Path(temporary.name) / "conversations.db")
        self.agent = Agent(Config.load(), self.store)
        object.__setattr__(
            self.agent._config,
            "working_notice_interval_seconds",
            0,
        )

    async def asyncTearDown(self) -> None:
        await self.agent.close()

    async def test_started_turn_is_cancelled_and_retired_as_stopped(self) -> None:
        model_started = asyncio.Event()

        async def blocked_stream(
            request,
            *,
            force_refresh,
            ttfb_timeout,
            idle_timeout,
        ):
            model_started.set()
            await asyncio.Future()

        self.agent._stream_once = blocked_stream
        response = asyncio.create_task(
            self.agent.respond_result("chat", "cancel this request")
        )
        await asyncio.wait_for(model_started.wait(), timeout=1)

        outcome = await asyncio.wait_for(self.agent.stop("chat"), timeout=0.5)

        self.assertEqual(outcome.status, StopStatus.STOPPED)
        self.assertEqual(outcome.previous_phase, "started")
        with self.assertRaises(TurnStopped) as stopped:
            await response
        self.assertEqual(stopped.exception.outcome, outcome)
        await self.agent.finalize_stop(outcome)
        self.assertEqual(
            self.store.load("chat", 2),
            [
                ("user", "cancel this request"),
                ("assistant", t("runtime.stopped", self.agent._config.language)),
            ],
        )

    async def test_preprocessing_stop_retires_claim_before_cancelling_child(self) -> None:
        child_started = asyncio.Event()
        child_cancelled = asyncio.Event()
        retired = asyncio.Event()

        async def before_stop() -> None:
            retired.set()

        async def blocked_preparation() -> None:
            child_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                child_cancelled.set()
                raise

        async def prepare() -> None:
            async with self.agent.prepare_turn(
                "chat",
                before_stop=before_stop,
            ) as execution:
                await self.agent.run_preparation_step(
                    "chat",
                    execution,
                    blocked_preparation,
                )

        owner = asyncio.create_task(prepare())
        await asyncio.wait_for(child_started.wait(), timeout=1)

        outcome = await self.agent.stop("chat")

        self.assertEqual(outcome.status, StopStatus.STOPPED)
        self.assertIsNone(outcome.checkpoint)
        self.assertTrue(retired.is_set())
        with self.assertRaises(TurnStopped):
            await owner
        await asyncio.wait_for(child_cancelled.wait(), timeout=1)
        self.assertEqual(self.store.list_active_turns(), [])
        self.assertNotIn("chat", self.agent._active_executions)

    async def test_answer_checkpoint_wins_a_concurrent_late_stop(self) -> None:
        async def answered_stream(
            request,
            *,
            force_refresh,
            ttfb_timeout,
            idle_timeout,
        ):
            return codex_stream.StreamResult(text="Natural answer.")

        entered = threading.Event()
        release = threading.Event()
        real_checkpoint = self.store.checkpoint_answer

        def delayed_checkpoint(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=1):
                raise TimeoutError("test did not release answer checkpoint")
            return real_checkpoint(*args, **kwargs)

        self.agent._stream_once = answered_stream
        with patch.object(self.store, "checkpoint_answer", delayed_checkpoint):
            response = asyncio.create_task(
                self.agent.respond_result(
                    "chat",
                    "answer normally",
                    defer_completion=True,
                )
            )
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            stop = asyncio.create_task(self.agent.stop("chat"))
            await asyncio.sleep(0)
            release.set()
            result = await response
            outcome = await stop

        self.assertEqual(result.text, "Natural answer.")
        self.assertEqual(outcome.status, StopStatus.TOO_LATE)
        self.assertEqual(outcome.previous_phase, "answer_ready")
        await self.agent.finalize_ready_turn("chat")
        self.assertEqual(
            self.store.load("chat", 2),
            [("user", "answer normally"), ("assistant", "Natural answer.")],
        )

    async def test_claim_retirement_and_finalize_precede_owner_release(self) -> None:
        model_started = asyncio.Event()
        retirement_started = asyncio.Event()
        allow_finalize = asyncio.Event()
        order: list[str] = []

        async def blocked_stream(
            request,
            *,
            force_refresh,
            ttfb_timeout,
            idle_timeout,
        ):
            model_started.set()
            await asyncio.Future()

        async def retire_claim() -> None:
            if not order:
                order.append("claim")

        async def handle() -> None:
            async with self.agent.prepare_turn(
                "chat",
                before_stop=retire_claim,
            ) as execution:
                try:
                    await self.agent.respond_result(
                        "chat",
                        "stop safely",
                        claim_ids=["a" * 64],
                        prepared_execution=execution,
                    )
                except TurnStopped as stopped:
                    order.append("claim")
                    retirement_started.set()
                    await allow_finalize.wait()
                    await self.agent.finalize_stop(stopped.outcome)
                    order.append("finalize")

        self.agent._stream_once = blocked_stream
        owner = asyncio.create_task(handle())
        await asyncio.wait_for(model_started.wait(), timeout=1)
        outcome = await self.agent.stop("chat")
        await asyncio.wait_for(retirement_started.wait(), timeout=1)

        self.assertFalse(await self.agent.forget("chat"))
        self.assertIs(self.agent._active_executions["chat"].owner_task, owner)
        self.assertEqual(self.store.list_active_turns()[0].phase, "stopped")

        allow_finalize.set()
        await owner
        self.assertTrue(await self.agent.forget("chat"))
        self.assertEqual(outcome.status, StopStatus.STOPPED)
        self.assertEqual(order, ["claim", "finalize"])
        self.assertNotIn("chat", self.agent._active_executions)
        self.assertEqual(self.store.list_active_turns(), [])

    async def test_external_owner_cancellation_cancels_the_model_child(self) -> None:
        model_started = asyncio.Event()
        model_cancelled = asyncio.Event()

        async def blocked_stream(
            request,
            *,
            force_refresh,
            ttfb_timeout,
            idle_timeout,
        ):
            model_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                model_cancelled.set()
                raise

        self.agent._stream_once = blocked_stream
        response = asyncio.create_task(self.agent.respond_result("chat", "cancel owner"))
        await asyncio.wait_for(model_started.wait(), timeout=1)
        response.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await response
        await asyncio.wait_for(model_cancelled.wait(), timeout=1)
        self.assertNotIn("chat", self.agent._active_executions)

    async def test_cancelled_stop_caller_does_not_cancel_shared_transition(self) -> None:
        model_started = asyncio.Event()
        stop_started = threading.Event()
        release_stop = threading.Event()
        real_request_stop = self.store.request_stop

        async def blocked_stream(
            request,
            *,
            force_refresh,
            ttfb_timeout,
            idle_timeout,
        ):
            model_started.set()
            await asyncio.Future()

        def delayed_stop(*args, **kwargs):
            stop_started.set()
            if not release_stop.wait(timeout=1):
                raise TimeoutError("test did not release stop transition")
            return real_request_stop(*args, **kwargs)

        self.agent._stream_once = blocked_stream
        response = asyncio.create_task(self.agent.respond_result("chat", "stop once"))
        await asyncio.wait_for(model_started.wait(), timeout=1)
        with patch.object(self.store, "request_stop", delayed_stop):
            first = asyncio.create_task(self.agent.stop("chat"))
            self.assertTrue(await asyncio.to_thread(stop_started.wait, 1))
            second = asyncio.create_task(self.agent.stop("chat"))
            await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            release_stop.set()
            outcome = await second

        self.assertEqual(outcome.status, StopStatus.STOPPED)
        with self.assertRaises(TurnStopped):
            await response
        await self.agent.finalize_stop(outcome)

    async def test_failed_stop_can_be_retried(self) -> None:
        model_started = asyncio.Event()
        real_request_stop = self.store.request_stop
        attempts = 0

        async def blocked_stream(
            request,
            *,
            force_refresh,
            ttfb_timeout,
            idle_timeout,
        ):
            model_started.set()
            await asyncio.Future()

        def flaky_stop(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConversationError("temporary persistence failure")
            return real_request_stop(*args, **kwargs)

        self.agent._stream_once = blocked_stream
        response = asyncio.create_task(self.agent.respond_result("chat", "retry stop"))
        await asyncio.wait_for(model_started.wait(), timeout=1)
        with patch.object(self.store, "request_stop", flaky_stop):
            with self.assertRaisesRegex(
                ConversationError,
                "temporary persistence failure",
            ):
                await self.agent.stop("chat")
            outcome = await self.agent.stop("chat")

        self.assertEqual(attempts, 2)
        self.assertEqual(outcome.status, StopStatus.STOPPED)
        with self.assertRaises(TurnStopped):
            await response
        await self.agent.finalize_stop(outcome)

    async def test_stop_cancels_the_single_progress_notice(self) -> None:
        model_started = asyncio.Event()
        notice_sent = asyncio.Event()
        notices: list[tuple[str, str]] = []

        async def blocked_stream(
            request,
            *,
            force_refresh,
            ttfb_timeout,
            idle_timeout,
        ):
            model_started.set()
            await asyncio.Future()

        async def notice(text: str, replace_id: str = "") -> SendResult:
            notices.append((text, replace_id))
            notice_sent.set()
            return SendResult(True, message_id="progress-1")

        object.__setattr__(
            self.agent._config,
            "working_notice_interval_seconds",
            0.01,
        )
        self.agent._stream_once = blocked_stream
        response = asyncio.create_task(
            self.agent.respond_result("chat", "long task", on_notice=notice)
        )
        await asyncio.wait_for(model_started.wait(), timeout=1)
        await asyncio.wait_for(notice_sent.wait(), timeout=1)

        outcome = await self.agent.stop("chat")
        with self.assertRaises(TurnStopped):
            await response
        count_after_stop = len(notices)
        await asyncio.sleep(0.05)

        self.assertEqual(outcome.status, StopStatus.STOPPED)
        self.assertEqual(len(notices), count_after_stop)
        self.assertEqual(notices[0][1], "")
        await self.agent.finalize_stop(outcome)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
