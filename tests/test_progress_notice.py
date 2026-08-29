from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from pilotage.agent import Agent
from pilotage.delivery import SendResult
from pilotage.main import _update_progress_notice


class _Approvals:
    @staticmethod
    def has_pending(_chat_id: str) -> bool:
        return False


def _notice_agent(*, language: str = "en", interval: float = 60.0) -> Agent:
    agent = object.__new__(Agent)
    agent._config = SimpleNamespace(
        language=language,
        working_notice_interval_seconds=interval,
        working_notice_text={
            "en": "Still working.",
            "fr": "Je continue.",
            "ar": "ما زلت أعمل.",
        }[language],
    )
    agent._approvals = _Approvals()
    return agent


class ProgressNoticeTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_message_is_then_edited_with_elapsed_minutes(self):
        agent = _notice_agent()
        calls: list[tuple[str, str]] = []
        sleeps = 0

        async def fake_sleep(_delay: float) -> None:
            nonlocal sleeps
            sleeps += 1
            if sleeps > 2:
                raise asyncio.CancelledError

        async def notice(text: str, replace_id: str = "") -> SendResult:
            calls.append((text, replace_id))
            return SendResult(True, message_id="progress-1")

        with (
            mock.patch("pilotage.agent.asyncio.sleep", side_effect=fake_sleep),
            mock.patch(
                "pilotage.agent.time.monotonic",
                side_effect=[0.0, 60.1, 120.1],
            ),
            self.assertRaises(asyncio.CancelledError),
        ):
            await agent._working_notice_loop("chat", notice)

        self.assertEqual(
            calls,
            [
                ("Still working. (1 min)", ""),
                ("Still working. (2 min)", "progress-1"),
            ],
        )

    async def test_sub_minute_duration_is_localized(self):
        expected = {
            "en": "Still working. (<1 min)",
            "fr": "Je continue. (<1 min)",
            "ar": "ما زلت أعمل. (أقل من دقيقة)",
        }
        for language, written in expected.items():
            with self.subTest(language=language):
                agent = _notice_agent(language=language, interval=5.0)
                seen: list[str] = []
                sleeps = 0

                async def fake_sleep(_delay: float) -> None:
                    nonlocal sleeps
                    sleeps += 1
                    if sleeps > 1:
                        raise asyncio.CancelledError

                async def notice(text: str, _replace_id: str = "") -> None:
                    seen.append(text)

                with (
                    mock.patch("pilotage.agent.asyncio.sleep", side_effect=fake_sleep),
                    mock.patch(
                        "pilotage.agent.time.monotonic",
                        side_effect=[0.0, 5.0],
                    ),
                    self.assertRaises(asyncio.CancelledError),
                ):
                    await agent._working_notice_loop("chat", notice)
                self.assertEqual(seen, [written])

    async def test_ambiguous_initial_send_stops_cosmetic_retries(self):
        agent = _notice_agent()
        calls = 0
        sleeps = 0

        async def fake_sleep(_delay: float) -> None:
            nonlocal sleeps
            sleeps += 1
            if sleeps > 2:
                raise asyncio.CancelledError

        async def notice(_text: str, _replace_id: str = "") -> SendResult:
            nonlocal calls
            calls += 1
            return SendResult(False, "ambiguous", retryable=False)

        with (
            mock.patch("pilotage.agent.asyncio.sleep", side_effect=fake_sleep),
            mock.patch(
                "pilotage.agent.time.monotonic",
                side_effect=[0.0, 60.1, 120.1],
            ),
        ):
            await agent._working_notice_loop("chat", notice)

        self.assertEqual(calls, 1)

    async def test_definitive_edit_rejection_never_sends_a_replacement(self):
        channel = SimpleNamespace(
            edit_message=mock.AsyncMock(
                return_value=SendResult(False, "unsupported", retryable=False)
            )
        )
        send = mock.AsyncMock(return_value=SendResult(True, message_id="new"))

        result = await _update_progress_notice(
            channel,
            "chat",
            "Still working. (2 min)",
            "progress-1",
            send=send,
        )

        self.assertFalse(result)
        channel.edit_message.assert_awaited_once_with(
            "chat",
            "progress-1",
            "Still working. (2 min)",
        )
        send.assert_not_awaited()

    async def test_success_without_message_id_is_attempted_only_once(self):
        agent = _notice_agent()
        calls = 0
        sleeps = 0

        async def fake_sleep(_delay: float) -> None:
            nonlocal sleeps
            sleeps += 1
            if sleeps > 2:
                raise asyncio.CancelledError

        async def notice(_text: str, _replace_id: str = "") -> SendResult:
            nonlocal calls
            calls += 1
            return SendResult(True)

        with (
            mock.patch("pilotage.agent.asyncio.sleep", side_effect=fake_sleep),
            mock.patch(
                "pilotage.agent.time.monotonic",
                side_effect=[0.0, 60.1, 120.1],
            ),
        ):
            await agent._working_notice_loop("chat", notice)

        self.assertEqual(calls, 1)

    async def test_retryable_edit_reuses_the_same_message_id(self):
        agent = _notice_agent()
        calls: list[str] = []
        sleeps = 0

        async def fake_sleep(_delay: float) -> None:
            nonlocal sleeps
            sleeps += 1
            if sleeps > 3:
                raise asyncio.CancelledError

        async def notice(_text: str, replace_id: str = "") -> SendResult:
            calls.append(replace_id)
            if not replace_id:
                return SendResult(True, message_id="progress-1")
            return SendResult(False, "retry", retryable=True)

        with (
            mock.patch("pilotage.agent.asyncio.sleep", side_effect=fake_sleep),
            mock.patch(
                "pilotage.agent.time.monotonic",
                side_effect=[0.0, 60.1, 120.1, 180.1],
            ),
            self.assertRaises(asyncio.CancelledError),
        ):
            await agent._working_notice_loop("chat", notice)

        self.assertEqual(calls, ["", "progress-1", "progress-1"])

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
