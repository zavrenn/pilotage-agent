"""The stream watchdog, against a connection that actually goes quiet.

The failure this guards against cannot be provoked on demand from the real
backend, so it is staged here: a stream that never speaks, and a stream that
says one thing and then stops forever. Both are the observed Codex failures.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from unittest import mock

import httpx
from openai import APIStatusError

from pilotage.agent import Agent
from pilotage.codex import auth
from pilotage.codex import stream as codex_stream
from pilotage.config import Config
from pilotage.history import ConversationStore
from pilotage.i18n import t

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
        self.assertTrue(result.terminal_completed)
        self.assertIsNotNone(result.timing)
        self.assertEqual(result.timing.event_count, 3)
        self.assertIsNotNone(result.timing.first_event_seconds)
        self.assertGreaterEqual(result.timing.elapsed_seconds, 0.0)
        self.assertGreaterEqual(result.timing.max_event_gap_seconds, 0.0)

    async def test_partial_text_without_a_terminal_event_is_not_completion_proof(self):
        result = await codex_stream.consume_stream(
            FakeStream([_delta("plausible partial")]),
            ttfb_timeout=1.0,
            idle_timeout=1.0,
        )

        self.assertEqual(result.text, "plausible partial")
        self.assertIs(result.terminal_completed, False)

    async def test_incomplete_terminal_status_is_not_completion_proof(self):
        incomplete = SimpleNamespace(
            type="response.incomplete",
            response=SimpleNamespace(
                status="incomplete", usage=None, id="resp_1", error=None
            ),
        )
        result = await codex_stream.consume_stream(
            FakeStream([_delta("partial"), incomplete]),
            ttfb_timeout=1.0,
            idle_timeout=1.0,
        )

        self.assertFalse(result.terminal_completed)

    async def test_completed_event_without_completed_status_is_not_proof(self):
        terminal = SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(status=None, usage=None, id="resp_1", error=None),
        )
        result = await codex_stream.consume_stream(
            FakeStream([_delta("unknown"), terminal]),
            ttfb_timeout=1.0,
            idle_timeout=1.0,
        )

        self.assertFalse(result.terminal_completed)

    async def test_a_stream_that_never_speaks_is_dropped(self):
        stream = FakeStream([HANG])
        with self.assertRaises(codex_stream.CodexStreamTimeout) as caught:
            await codex_stream.consume_stream(stream, ttfb_timeout=0.05, idle_timeout=0.05)
        self.assertEqual(caught.exception.code, "codex_stream_no_first_byte")
        self.assertIsNotNone(caught.exception.timing)
        self.assertEqual(caught.exception.timing.event_count, 0)
        self.assertIsNone(caught.exception.timing.first_event_seconds)
        self.assertGreaterEqual(caught.exception.timing.last_event_gap_seconds, 0.04)

    async def test_a_stream_that_stops_mid_answer_is_dropped(self):
        stream = FakeStream([_delta("Half an ans"), HANG])
        with self.assertRaises(codex_stream.CodexStreamTimeout) as caught:
            await codex_stream.consume_stream(stream, ttfb_timeout=5.0, idle_timeout=0.05)
        self.assertEqual(caught.exception.code, "codex_stream_stalled")
        self.assertIsNotNone(caught.exception.timing)
        self.assertEqual(caught.exception.timing.event_count, 1)
        self.assertIsNotNone(caught.exception.timing.first_event_seconds)
        self.assertGreaterEqual(caught.exception.timing.last_event_gap_seconds, 0.04)

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

    def test_a_long_conversation_only_extends_idle_patience(self):
        # Roughly 75,000 tokens of conversation.
        request = {
            "instructions": "Be brief.",
            "input": [{"role": "user", "content": "x" * 300_000}],
        }
        ttfb, idle = codex_stream.stream_timeouts(
            request, ttfb_timeout=120.0, idle_timeout=12.0
        )
        self.assertEqual(idle, 120.0)
        self.assertEqual(ttfb, 120.0)

    def test_text_is_still_counted(self):
        request = {"instructions": "", "input": [{"role": "user", "content": "y" * 80_000}]}
        estimate = codex_stream.estimate_context_tokens(request)
        self.assertAlmostEqual(estimate, 20_000, delta=100)

    def test_tool_schemas_count_toward_request_patience(self):
        request = {
            "instructions": "Be brief.",
            "input": [{"role": "user", "content": "Use the available tool."}],
            "tools": [
                {
                    "type": "function",
                    "name": "large_tool",
                    "description": "x" * 40_000,
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                }
            ],
        }

        estimate = codex_stream.estimate_context_tokens(request)
        ttfb, idle = codex_stream.stream_timeouts(
            request,
            ttfb_timeout=120.0,
            idle_timeout=12.0,
        )

        self.assertGreater(estimate, 10_000)
        self.assertEqual(idle, 60.0)
        self.assertEqual(ttfb, 120.0)

    def test_context_size_does_not_multiply_the_first_event_budget(self):
        request = {"instructions": "", "input": [{"role": "user", "content": "x" * 8_000_000}]}
        ttfb, _ = codex_stream.stream_timeouts(request, ttfb_timeout=120.0, idle_timeout=12.0)
        self.assertEqual(ttfb, 120.0)

    def test_an_explicit_first_event_budget_is_preserved(self):
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


def _bad_request(*, code: str, message: str) -> APIStatusError:
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
    )
    return APIStatusError(
        message,
        response=response,
        body={"code": code, "message": message},
    )


class ReconnectTests(unittest.IsolatedAsyncioTestCase):
    """One quiet connection is retried; a second one is the backend, not the socket."""

    def setUp(self):
        # Nothing here touches history, but an Agent writes one; keep it out of
        # the real state directory.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.agent = Agent(Config.load(), ConversationStore(Path(tmp.name) / "conversations.db"))
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
        self.assertEqual(self.notices, [])

    async def test_successful_stream_logs_non_sensitive_timing(self):
        async def stream_once(_request, **_kwargs):
            return codex_stream.StreamResult(
                text="private answer",
                status="completed",
                terminal_completed=True,
                timing=codex_stream.StreamTiming(
                    elapsed_seconds=3.25,
                    first_event_seconds=0.5,
                    event_count=7,
                    max_event_gap_seconds=1.25,
                    last_event_gap_seconds=0.0,
                ),
            )

        self.agent._stream_once = stream_once
        with self.assertLogs("pilotage.agent", level="INFO") as captured:
            await self.agent._call_model(
                "chat",
                [{"role": "user", "content": "private prompt"}],
                None,
            )

        written = "\n".join(captured.output)
        self.assertIn("elapsed=3.25s", written)
        self.assertIn("first_event=0.50s", written)
        self.assertIn("events=7", written)
        self.assertIn("max_event_gap=1.25s", written)
        self.assertNotIn("private prompt", written)
        self.assertNotIn("private answer", written)

    async def test_a_second_quiet_connection_gives_up(self):
        quiet = [
            codex_stream.CodexStreamTimeout("quiet", code="codex_stream_stalled"),
            codex_stream.CodexStreamTimeout("quiet again", code="codex_stream_stalled"),
        ]
        self.agent._stream_once = self._answer_after(quiet)
        with self.assertLogs("pilotage.agent", level="WARNING"):
            result = await self.agent._run_turn("chat", "hi", [], self._notice)
        self.assertEqual(result.text, t("runtime.failure", self.agent._config.language))
        self.assertFalse(result.terminal_completed)
        self.assertEqual(len(self.attempts), 2)
        self.assertEqual(self.notices, [])

    async def test_a_transport_interruption_is_retried_once(self):
        broken = httpx.ReadError(
            "connection reset",
            request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
        )
        self.agent._stream_once = self._answer_after([broken])

        result = await self.agent._run_turn("chat", "hi", [], self._notice)

        self.assertEqual(result.text, "Answered.")
        self.assertEqual(len(self.attempts), 2)
        self.assertEqual(self.notices, [])

    async def test_transport_and_silence_share_one_reconnect_budget(self):
        failures = [
            codex_stream.CodexStreamTimeout(
                "quiet", code="codex_stream_stalled"
            ),
            httpx.ReadError(
                "connection reset",
                request=httpx.Request(
                    "POST", "https://chatgpt.com/backend-api/codex/responses"
                ),
            ),
        ]
        self.agent._stream_once = self._answer_after(failures)

        with self.assertLogs("pilotage.agent", level="WARNING"):
            result = await self.agent._run_turn("chat", "hi", [], self._notice)

        self.assertEqual(result.text, t("runtime.failure", self.agent._config.language))
        self.assertFalse(result.terminal_completed)
        self.assertEqual(len(self.attempts), 2)
        self.assertEqual(self.notices, [])

    async def test_an_api_status_failure_becomes_a_local_failure_reply(self):
        self.agent._stream_once = self._answer_after(
            [
                _bad_request(
                    code="content_policy_violation",
                    message="Request blocked.",
                )
            ]
        )

        with self.assertLogs("pilotage.agent", level="WARNING"):
            result = await self.agent._run_turn("chat", "hi", [], self._notice)

        self.assertEqual(result.text, t("runtime.failure", self.agent._config.language))
        self.assertFalse(result.terminal_completed)
        self.assertEqual(len(self.attempts), 1)

    async def test_a_reconnect_does_not_spend_a_refresh_token(self):
        """The credentials are not what went quiet; refreshing would rotate them for nothing."""
        quiet = codex_stream.CodexStreamTimeout("quiet", code="codex_stream_stalled")
        self.agent._stream_once = self._answer_after([_unauthorized(), quiet])
        result = await self.agent._run_turn("chat", "hi", [], self._notice)
        self.assertEqual(result.text, "Answered.")
        # Refresh once for the 401, then leave the fresh credentials alone.
        self.assertEqual(self.attempts, [False, True, False])

    async def test_a_reconnect_does_not_call_the_public_notice(self):
        async def explode(text: str) -> None:
            raise AssertionError("an internal reconnect reached the channel")

        quiet = codex_stream.CodexStreamTimeout("quiet", code="codex_stream_stalled")
        self.agent._stream_once = self._answer_after([quiet])
        result = await self.agent._run_turn("chat", "hi", [], explode)
        self.assertEqual(result.text, "Answered.")

    async def test_exact_masked_replay_rejection_strips_opaque_items_once(self):
        requests = []
        rejection = _bad_request(
            code="invalid_prompt",
            message="Request blocked.",
        )

        async def stream_once(
            request, *, force_refresh, ttfb_timeout, idle_timeout
        ):
            requests.append(request)
            if len(requests) == 1:
                raise rejection
            return codex_stream.StreamResult(text="Recovered.")

        self.agent._stream_once = stream_once
        result = await self.agent._call_model(
            "chat",
            [
                {"type": "reasoning", "encrypted_content": "reasoning"},
                {"type": "compaction", "encrypted_content": "checkpoint"},
                {"role": "user", "content": "continue"},
            ],
            None,
        )

        self.assertEqual(result.text, "Recovered.")
        self.assertTrue(
            any(item.get("encrypted_content") for item in requests[0]["input"])
        )
        self.assertFalse(
            any(item.get("encrypted_content") for item in requests[1]["input"])
        )
        self.assertIn(
            {"role": "user", "content": "continue"},
            requests[1]["input"],
        )

    async def test_masked_rejection_without_opaque_replay_is_not_retried(self):
        calls = 0

        async def reject(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise _bad_request(
                code="invalid_prompt",
                message="Request blocked.",
            )

        self.agent._stream_once = reject
        with self.assertRaises(APIStatusError):
            await self.agent._call_model(
                "chat",
                [{"role": "user", "content": "ordinary prompt"}],
                None,
            )
        self.assertEqual(calls, 1)

    async def test_ordinary_content_block_is_not_reclassified_as_stale_replay(self):
        calls = 0

        async def reject(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise _bad_request(
                code="content_policy_violation",
                message="Request blocked.",
            )

        self.agent._stream_once = reject
        with self.assertRaises(APIStatusError):
            await self.agent._call_model(
                "chat",
                [
                    {"type": "reasoning", "encrypted_content": "opaque"},
                    {"role": "user", "content": "request"},
                ],
                None,
            )
        self.assertEqual(calls, 1)

    async def test_masked_replay_repair_never_loops(self):
        requests = []

        async def reject(request, **_kwargs):
            requests.append(request)
            raise _bad_request(
                code="invalid_prompt",
                message="Request blocked.",
            )

        self.agent._stream_once = reject
        with self.assertRaises(APIStatusError):
            await self.agent._call_model(
                "chat",
                [
                    {"type": "reasoning", "encrypted_content": "opaque"},
                    {"role": "user", "content": "request"},
                ],
                None,
            )

        self.assertEqual(len(requests), 2)
        self.assertFalse(
            any(item.get("encrypted_content") for item in requests[-1]["input"])
        )


class ClientLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.agent = Agent(
            Config.load(),
            ConversationStore(Path(tmp.name) / "conversations.db"),
        )

    async def test_stream_creation_itself_is_bounded_by_ttfb(self):
        class HangingResponses:
            async def create(self, **_kwargs):
                await asyncio.Event().wait()

        async def client(*, force_refresh=False):
            return SimpleNamespace(responses=HangingResponses())

        self.agent._ensure_client = client
        with self.assertRaises(codex_stream.CodexStreamTimeout) as caught:
            await self.agent._stream_once(
                {"model": "gpt-test", "input": []},
                force_refresh=False,
                ttfb_timeout=0.02,
                idle_timeout=1.0,
            )
        self.assertEqual(caught.exception.code, "codex_stream_no_first_byte")

    async def test_creation_and_first_event_share_one_ttfb_budget(self):
        class DelayedResponses:
            async def create(self, **_kwargs):
                await asyncio.sleep(0.1)
                return FakeStream([0.1, _completed()])

        async def client(*, force_refresh=False):
            return SimpleNamespace(responses=DelayedResponses())

        self.agent._ensure_client = client
        started_at = asyncio.get_running_loop().time()
        with self.assertRaises(codex_stream.CodexStreamTimeout) as caught:
            await self.agent._stream_once(
                {"model": "gpt-test", "input": []},
                force_refresh=False,
                ttfb_timeout=0.15,
                idle_timeout=1.0,
            )
        elapsed = asyncio.get_running_loop().time() - started_at

        self.assertEqual(caught.exception.code, "codex_stream_no_first_byte")
        self.assertIsNotNone(caught.exception.timing)
        self.assertGreaterEqual(caught.exception.timing.elapsed_seconds, 0.09)
        self.assertLess(elapsed, 0.19)
        self.assertTrue(caught.exception.timing.first_event_seconds is None)

    async def test_success_timing_includes_stream_creation(self):
        class DelayedResponses:
            async def create(self, **_kwargs):
                await asyncio.sleep(0.02)
                return FakeStream([_completed()])

        async def client(*, force_refresh=False):
            return SimpleNamespace(responses=DelayedResponses())

        self.agent._ensure_client = client
        result = await self.agent._stream_once(
            {"model": "gpt-test", "input": []},
            force_refresh=False,
            ttfb_timeout=1.0,
            idle_timeout=1.0,
        )

        self.assertIsNotNone(result.timing)
        self.assertGreaterEqual(result.timing.first_event_seconds, 0.015)

    async def test_stream_call_routes_bulk_payload_around_the_sdk_transform(self):
        class RecordingResponses:
            def __init__(self):
                self.kwargs = None

            async def create(self, **kwargs):
                self.kwargs = kwargs
                return FakeStream([_completed()])

        responses = RecordingResponses()

        async def client(*, force_refresh=False):
            return SimpleNamespace(responses=responses)

        request = {
            "model": "gpt-test",
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "name": "terminal", "parameters": {}}],
        }
        self.agent._ensure_client = client

        await self.agent._stream_once(
            request,
            force_refresh=False,
            ttfb_timeout=1.0,
            idle_timeout=1.0,
        )

        self.assertNotIn("input", responses.kwargs)
        self.assertNotIn("tools", responses.kwargs)
        self.assertEqual(responses.kwargs["extra_body"]["input"], request["input"])
        self.assertEqual(responses.kwargs["extra_body"]["tools"], request["tools"])
        self.assertTrue(responses.kwargs["stream"])

    async def test_close_releases_current_and_retired_clients_once(self):
        class FakeClient:
            def __init__(self):
                self.closed = 0

            async def close(self):
                self.closed += 1

        retired = FakeClient()
        current = FakeClient()
        self.agent._retired_clients[id(retired)] = retired
        self.agent._client = current

        await self.agent.close()
        await self.agent.close()

        self.assertEqual((retired.closed, current.closed), (1, 1))

    async def test_replaced_client_closes_when_its_last_stream_releases_it(self):
        class FakeClient:
            def __init__(self):
                self.closed = 0

            async def close(self):
                self.closed += 1

        old = FakeClient()
        replacement = FakeClient()
        old_credentials = auth.Credentials("old", "refresh", "https://codex", "")
        new_credentials = auth.Credentials("new", "refresh", "https://codex", "")
        self.agent._client = old
        self.agent._credentials = old_credentials

        with mock.patch.object(
            auth, "access_token_is_expiring", return_value=False
        ):
            borrowed_old = await self.agent._ensure_client()
        self.assertIs(borrowed_old, old)

        with (
            mock.patch.object(
                auth, "resolve_credentials", return_value=new_credentials
            ),
            mock.patch(
                "pilotage.agent.codex_client.build_client",
                return_value=replacement,
            ),
        ):
            borrowed = await self.agent._ensure_client(force_refresh=True)

        self.assertIs(borrowed, replacement)
        self.assertEqual(old.closed, 0)
        self.assertEqual(list(self.agent._retired_clients.values()), [old])

        await self.agent._release_client(old)
        self.assertEqual(old.closed, 1)
        self.assertEqual(self.agent._retired_clients, {})

        await self.agent._release_client(replacement)
        await self.agent.close()

    async def test_idle_replaced_client_does_not_accumulate(self):
        class FakeClient:
            def __init__(self):
                self.closed = 0

            async def close(self):
                self.closed += 1

        old = FakeClient()
        replacement = FakeClient()
        self.agent._client = old
        self.agent._credentials = auth.Credentials(
            "old", "refresh", "https://codex", ""
        )
        refreshed = auth.Credentials("new", "refresh", "https://codex", "")

        with (
            mock.patch.object(auth, "resolve_credentials", return_value=refreshed),
            mock.patch(
                "pilotage.agent.codex_client.build_client",
                return_value=replacement,
            ),
        ):
            borrowed = await self.agent._ensure_client(force_refresh=True)

        self.assertIs(borrowed, replacement)
        self.assertEqual(old.closed, 1)
        self.assertEqual(self.agent._retired_clients, {})

        await self.agent._release_client(replacement)
        await self.agent.close()


if __name__ == "__main__":
    unittest.main()
