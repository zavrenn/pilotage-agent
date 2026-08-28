"""Starting a conversation over.

The command has to do two things a plain message never does: skip the model
entirely, and clear history that a turn already running is about to write back.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from pilotage.agent import Agent
from pilotage.channels.whatsapp import RESET_COMMAND, InboundMessage, WhatsAppChannel
from pilotage.codex import stream as codex_stream
from pilotage.config import Config
from pilotage.history import ConversationError, ConversationStore


def _event(text: str, chat_id: str = "chat", message_id: str = "m1") -> Dict[str, Any]:
    return {
        "chatId": chat_id,
        "messageId": message_id,
        "_pilotageClaimId": hashlib.sha256(
            f"{chat_id}|{message_id}".encode("utf-8")
        ).hexdigest(),
        "senderId": "212600000000@s.whatsapp.net",
        "senderNumber": "212600000000",
        "body": text,
    }


class ChannelCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = Config.load()
        object.__setattr__(config, "state_dir", Path(temporary.name))
        object.__setattr__(config, "allowed_senders", frozenset({"212600000000"}))
        object.__setattr__(config, "text_batch_delay_seconds", 30.0)
        self.answered: List[InboundMessage] = []
        self.reset_for: List[tuple[str, str, str]] = []

        async def handler(message: InboundMessage) -> None:
            self.answered.append(message)

        async def on_command(
            chat_id: str,
            session_id: str,
            message_id: str,
            _invocation,
            claim_id: str,
        ) -> None:
            self.reset_for.append((chat_id, session_id, message_id))
            self.assertRegex(claim_id, r"^[a-f0-9]{64}$")

        self.channel = WhatsAppChannel(config, handler, on_command)

    async def _settle(self) -> None:
        """Let the detached command task run."""
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def test_the_command_resets_and_never_reaches_the_model(self):
        self.channel._accept(_event(RESET_COMMAND))
        await self._settle()
        self.assertEqual(self.reset_for, [("chat", "chat", "m1")])
        self.assertEqual(self.answered, [])

    async def test_case_and_stray_spaces_still_count(self):
        self.channel._accept(_event("  /NEW  "))
        await self._settle()
        self.assertEqual(self.reset_for, [("chat", "chat", "m1")])

    async def test_reset_alias_uses_the_same_command(self):
        self.channel._accept(_event("/reset"))
        await self._settle()
        self.assertEqual(self.reset_for, [("chat", "chat", "m1")])


    async def test_the_word_inside_a_sentence_is_just_a_message(self):
        self.channel._accept(_event("what is /new for?"))
        await self._settle()
        self.assertEqual(self.reset_for, [])
        self.assertEqual(len(self.channel._pending), 1)

    async def test_a_half_typed_question_is_dropped_with_it(self):
        """The batch is still waiting; sending it would answer the ended chat."""
        old_event = _event("actually, about yesterday", message_id="m1")
        old_claim = old_event["_pilotageClaimId"]
        self.channel._accept(old_event)
        self.assertEqual(len(self.channel._pending), 1)
        with mock.patch.object(self.channel, "_ack_later") as acknowledge:
            self.channel._accept(_event(RESET_COMMAND, message_id="m2"))
            # The reset boundary is durable before the bridge is allowed to
            # remove the superseded input from its own queue.
            self.assertTrue(self.channel._completed_claim_store.contains(old_claim))
            acknowledge.assert_called_once_with([old_claim])
        await self._settle()
        self.assertEqual(self.channel._pending, {})
        self.assertEqual(self.answered, [])
        # The confirmation quotes the command, not the abandoned question.
        self.assertEqual(self.reset_for, [("chat", "chat", "m2")])

    async def test_failed_durable_reset_boundary_stops_before_ack_or_command(self):
        self.channel._accept(_event("unfinished", message_id="m1"))

        with (
            mock.patch.object(
                self.channel._completed_claim_store,
                "mark",
                side_effect=OSError("disk unavailable"),
            ),
            mock.patch.object(self.channel, "_ack_later") as acknowledge,
        ):
            self.channel._accept(_event(RESET_COMMAND, message_id="m2"))
            await self._settle()

        self.assertEqual(
            self.channel.failure,
            "The WhatsApp completed-message ledger failed.",
        )
        acknowledge.assert_not_called()
        self.assertEqual(self.reset_for, [])


class ForgetTests(unittest.IsolatedAsyncioTestCase):
    """Clearing history is not just popping a dict — a turn may be mid-flight."""

    def setUp(self):
        # A store of its own: a test must never write into the real agent's
        # conversations, and "/new" here is a real write.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.agent = Agent(Config.load(), ConversationStore(Path(tmp.name) / "conversations.db"))

    async def test_history_is_cleared(self):
        open_gate = asyncio.Event()
        open_gate.set()
        self.agent._stream_once = self._answering(open_gate)
        await self.agent.respond("chat", "remember this")
        self.assertIn("chat", self.agent._history)
        await self.agent.forget("chat")
        self.assertNotIn("chat", self.agent._history)

    async def test_failed_durable_reset_keeps_the_live_conversation(self):
        gate = asyncio.Event()
        gate.set()
        self.agent._stream_once = self._answering(gate)
        await self.agent.respond("chat", "remember this")
        self.agent._tool_state["chat"] = {"todo": ["keep"]}
        self.agent._session_instructions["chat"] = "keep instructions"

        def _fail(_chat_id: str) -> None:
            raise ConversationError("disk unavailable")

        self.agent._store.new_session = _fail
        with self.assertRaisesRegex(ConversationError, "disk unavailable"):
            await self.agent.forget("chat")

        self.assertIn("chat", self.agent._history)
        self.assertEqual(self.agent._tool_state["chat"], {"todo": ["keep"]})
        self.assertEqual(
            self.agent._session_instructions["chat"], "keep instructions"
        )


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
