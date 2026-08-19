"""Starting a conversation over.

The command has to do two things a plain message never does: skip the model
entirely, and clear history that a turn already running is about to write back.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any, Dict, List

from pilotage.agent import Agent
from pilotage.channels.whatsapp import RESET_COMMAND, InboundMessage, WhatsAppChannel
from pilotage.codex import stream as codex_stream
from pilotage.config import Config


def _event(text: str, chat_id: str = "chat", message_id: str = "m1") -> Dict[str, Any]:
    return {
        "chatId": chat_id,
        "messageId": message_id,
        "senderId": "212600000000@s.whatsapp.net",
        "senderNumber": "212600000000",
        "body": text,
    }


class ChannelCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        config = Config.load()
        object.__setattr__(config, "allowed_senders", frozenset({"212600000000"}))
        object.__setattr__(config, "text_batch_delay_seconds", 30.0)
        self.answered: List[InboundMessage] = []
        self.reset_for: List[str] = []

        async def handler(message: InboundMessage) -> None:
            self.answered.append(message)

        async def on_reset(chat_id: str) -> None:
            self.reset_for.append(chat_id)

        self.channel = WhatsAppChannel(config, handler, on_reset)

    async def _settle(self) -> None:
        """Let the detached reset task run."""
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def test_the_command_resets_and_never_reaches_the_model(self):
        self.channel._accept(_event(RESET_COMMAND))
        await self._settle()
        self.assertEqual(self.reset_for, ["chat"])
        self.assertEqual(self.answered, [])

    async def test_case_and_stray_spaces_still_count(self):
        self.channel._accept(_event("  /NEW  "))
        await self._settle()
        self.assertEqual(self.reset_for, ["chat"])

    async def test_the_word_inside_a_sentence_is_just_a_message(self):
        self.channel._accept(_event("what is /new for?"))
        await self._settle()
        self.assertEqual(self.reset_for, [])
        self.assertEqual(len(self.channel._pending), 1)

    async def test_a_half_typed_question_is_dropped_with_it(self):
        """The batch is still waiting; sending it would answer the ended chat."""
        self.channel._accept(_event("actually, about yesterday", message_id="m1"))
        self.assertEqual(len(self.channel._pending), 1)
        self.channel._accept(_event(RESET_COMMAND, message_id="m2"))
        await self._settle()
        self.assertEqual(self.channel._pending, {})
        self.assertEqual(self.answered, [])


class ForgetTests(unittest.IsolatedAsyncioTestCase):
    """Clearing history is not just popping a dict — a turn may be mid-flight."""

    def setUp(self):
        self.agent = Agent(Config.load())

    async def test_history_is_cleared(self):
        open_gate = asyncio.Event()
        open_gate.set()
        self.agent._stream_once = self._answering(open_gate)
        await self.agent.respond("chat", "remember this")
        self.assertIn("chat", self.agent._history)
        await self.agent.forget("chat")
        self.assertNotIn("chat", self.agent._history)

    def _answering(self, gate: asyncio.Event):
        async def _stream_once(request, *, force_refresh, ttfb_timeout, idle_timeout):
            if not gate.is_set():
                await gate.wait()
            return codex_stream.StreamResult(text="Answered.")

        return _stream_once

    async def test_a_turn_in_flight_does_not_survive_the_reset(self):
        gate = asyncio.Event()
        self.agent._stream_once = self._answering(gate)
        turn = asyncio.create_task(self.agent.respond("chat", "the old question"))
        await asyncio.sleep(0)  # let the turn take the chat's lock

        forgetting = asyncio.create_task(self.agent.forget("chat"))
        await asyncio.sleep(0)
        gate.set()
        await turn
        await forgetting

        self.assertNotIn("chat", self.agent._history)


if __name__ == "__main__":
    unittest.main()
