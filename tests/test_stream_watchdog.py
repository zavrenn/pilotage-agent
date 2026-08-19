"""The stream watchdog, against a connection that actually goes quiet.

The failure this guards against cannot be provoked on demand from the real
backend, so it is staged here: a stream that never speaks, and a stream that
says one thing and then stops forever. Both are the observed Codex failures.
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any, List

import httpx
from openai import APIStatusError

from pilotage import agent as agent_module
from pilotage.agent import Agent
from pilotage.codex import stream as codex_stream
from pilotage.config import Config

# A step in a staged stream that never finishes.
HANG = object()


class FakeStream:
    """An async iterator that follows a script of events, pauses and hangs."""

    def __init__(self, script: List[Any]):
        self._script = list(script)
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for step in self._script:
            if step is HANG:
                await asyncio.Event().wait()  # never set: the socket that stays open
            elif isinstance(step, float):
                await asyncio.sleep(step)
            else:
                yield step

    async def close(self) -> None:
        self.closed = True


def _delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def _completed() -> SimpleNamespace:
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(status="completed", usage=None, id="resp_1", error=None),
    )


class ConsumeStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_healthy_stream_returns_its_text(self):
        stream = FakeStream([_delta("Hello "), _delta("there."), _completed()])
        result = await codex_stream.consume_stream(stream, ttfb_timeout=1.0, idle_timeout=1.0)
        self.assertEqual(result.text, "Hello there.")
        self.assertEqual(result.status, "completed")

    async def test_a_stream_that_never_speaks_is_dropped(self):
        stream = FakeStream([HANG])
        with self.assertRaises(codex_stream.CodexStreamTimeout) as caught:
            await codex_stream.consume_stream(stream, ttfb_timeout=0.05, idle_timeout=0.05)
        self.assertEqual(caught.exception.code, "codex_stream_no_first_byte")

    async def test_a_stream_that_stops_mid_answer_is_dropped(self):
        stream = FakeStream([_delta("Half an ans"), HANG])
        with self.assertRaises(codex_stream.CodexStreamTimeout) as caught:
            await codex_stream.consume_stream(stream, ttfb_timeout=5.0, idle_timeout=0.05)
        self.assertEqual(caught.exception.code, "codex_stream_stalled")

    async def test_the_two_cutoffs_are_told_apart(self):
        """A slow start is not a stall: the long wait applies before the first event."""
        stream = FakeStream([0.15, _delta("Late but fine."), _completed()])
        result = await codex_stream.consume_stream(stream, ttfb_timeout=5.0, idle_timeout=0.05)
        self.assertEqual(result.text, "Late but fine.")

    async def test_a_zero_cutoff_waits(self):
        stream = FakeStream([0.15, _delta("Slow."), 0.15, _completed()])
        result = await codex_stream.consume_stream(stream, ttfb_timeout=0, idle_timeout=0)
        self.assertEqual(result.text, "Slow.")

    async def test_an_error_event_still_wins_over_the_watchdog(self):
        stream = FakeStream([SimpleNamespace(type="error", message="nope", code="bad", param="")])
        with self.assertRaises(codex_stream.CodexStreamError) as caught:
            await codex_stream.consume_stream(stream, ttfb_timeout=1.0, idle_timeout=1.0)
        self.assertNotIsInstance(caught.exception, codex_stream.CodexStreamTimeout)


class ContextSizingTests(unittest.TestCase):
    """A picture must not read as a very long conversation."""

    def _photo_request(self, count: int = 1) -> dict:
        parts: List[dict] = [{"type": "input_text", "text": "What is this?"}]
        for _ in range(count):
            # Roughly a 3 MB photo once base64 encoded.
            parts.append(
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + "A" * 4_000_000}
            )
        return {"instructions": "Be brief.", "input": [{"role": "user", "content": parts}]}

    def test_a_photo_is_not_counted_as_its_base64(self):
        estimate = codex_stream.estimate_context_tokens(self._photo_request())
        self.assertLess(estimate, 10_000)

    def test_a_photo_keeps_the_short_quiet_cutoff(self):
        ttfb, idle = codex_stream.stream_timeouts(
            self._photo_request(), ttfb_timeout=120.0, idle_timeout=12.0
        )
        self.assertEqual(idle, 12.0)
        self.assertEqual(ttfb, 120.0)

    def test_several_photos_still_stay_small(self):
        estimate = codex_stream.estimate_context_tokens(self._photo_request(count=4))
        self.assertLess(estimate, 10_000)

    def test_a_long_conversation_earns_more_patience(self):
        # Roughly 75,000 tokens of conversation.
        request = {
            "instructions": "Be brief.",
            "input": [{"role": "user", "content": "x" * 300_000}],
        }
        ttfb, idle = codex_stream.stream_timeouts(
            request, ttfb_timeout=120.0, idle_timeout=12.0
        )
        self.assertEqual(idle, 120.0)
        # The first event is what a big prefill delays, so that wait grows too.
        self.assertGreater(ttfb, 120.0)

    def test_text_is_still_counted(self):
        request = {"instructions": "", "input": [{"role": "user", "content": "y" * 80_000}]}
        estimate = codex_stream.estimate_context_tokens(request)
        self.assertAlmostEqual(estimate, 20_000, delta=100)

    def test_the_first_event_wait_is_capped(self):
        """Nobody waits out nine minutes of silence, however big the request."""
        request = {"instructions": "", "input": [{"role": "user", "content": "x" * 8_000_000}]}
        ttfb, _ = codex_stream.stream_timeouts(request, ttfb_timeout=120.0, idle_timeout=12.0)
        self.assertEqual(ttfb, codex_stream.MAX_FIRST_EVENT_TIMEOUT_SECONDS)

    def test_an_explicit_setting_is_never_capped(self):
        """The cap bounds what we add, not what was asked for."""
        request = {"instructions": "", "input": [{"role": "user", "content": "x" * 8_000_000}]}
        ttfb, _ = codex_stream.stream_timeouts(request, ttfb_timeout=600.0, idle_timeout=12.0)
        self.assertEqual(ttfb, 600.0)

    def test_a_disabled_watchdog_stays_disabled(self):
        request = {"instructions": "", "input": [{"role": "user", "content": "z" * 800_000}]}
        ttfb, idle = codex_stream.stream_timeouts(request, ttfb_timeout=0, idle_timeout=0)
        self.assertEqual((ttfb, idle), (0, 0))


def _unauthorized() -> APIStatusError:
    response = httpx.Response(401, request=httpx.Request("POST", "https://example.invalid"))
    return APIStatusError("unauthorized", response=response, body=None)


class ReconnectTests(unittest.IsolatedAsyncioTestCase):
    """One quiet connection is retried; a second one is the backend, not the socket."""

    def setUp(self):
        self.agent = Agent(Config.load())
        self.attempts: List[bool] = []
        self.notices: List[str] = []

    async def _notice(self, text: str) -> None:
        self.notices.append(text)

    def _answer_after(self, failures: List[Any]):
        pending = list(failures)

        async def _stream_once(request, *, force_refresh, ttfb_timeout, idle_timeout):
            self.attempts.append(force_refresh)
            if pending:
                raise pending.pop(0)
            return codex_stream.StreamResult(text="Answered.")

        return _stream_once

    async def test_a_quiet_connection_is_retried_once(self):
        quiet = codex_stream.CodexStreamTimeout("quiet", code="codex_stream_stalled")
        self.agent._stream_once = self._answer_after([quiet])
        result = await self.agent._run_turn("chat", "hi", [], self._notice)
        self.assertEqual(result.text, "Answered.")
        self.assertEqual(len(self.attempts), 2)
        self.assertEqual(self.notices, [agent_module.RECONNECT_NOTICE])

    async def test_a_second_quiet_connection_gives_up(self):
        quiet = [
            codex_stream.CodexStreamTimeout("quiet", code="codex_stream_stalled"),
            codex_stream.CodexStreamTimeout("quiet again", code="codex_stream_stalled"),
        ]
        self.agent._stream_once = self._answer_after(quiet)
        with self.assertRaises(codex_stream.CodexStreamTimeout):
            await self.agent._run_turn("chat", "hi", [], self._notice)
        self.assertEqual(len(self.attempts), 2)

    async def test_a_reconnect_does_not_spend_a_refresh_token(self):
        """The credentials are not what went quiet; refreshing would rotate them for nothing."""
        quiet = codex_stream.CodexStreamTimeout("quiet", code="codex_stream_stalled")
        self.agent._stream_once = self._answer_after([_unauthorized(), quiet])
        result = await self.agent._run_turn("chat", "hi", [], self._notice)
        self.assertEqual(result.text, "Answered.")
        # Refresh once for the 401, then leave the fresh credentials alone.
        self.assertEqual(self.attempts, [False, True, False])

    async def test_a_failed_notice_does_not_fail_the_turn(self):
        async def explode(text: str) -> None:
            raise RuntimeError("the phone is unreachable")

        quiet = codex_stream.CodexStreamTimeout("quiet", code="codex_stream_stalled")
        self.agent._stream_once = self._answer_after([quiet])
        result = await self.agent._run_turn("chat", "hi", [], explode)
        self.assertEqual(result.text, "Answered.")


if __name__ == "__main__":
    unittest.main()
