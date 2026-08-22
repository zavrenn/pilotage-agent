"""Quoting the message an answer belongs to.

One agent answers several people, and an answer can arrive a minute after the
question that earned it. Unattached, it reads as a reply to whatever happens to
be above it — which in a group, or after a burst of three messages, is usually
the wrong thing.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any, Dict, List

from pilotage.channels.whatsapp import InboundMessage, WhatsAppChannel
from pilotage.config import Config


class FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        pass


class FakeHttp:
    """Stands in for the bridge and keeps what was posted to it."""

    def __init__(self) -> None:
        self.posts: List[Dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.posts.append(kwargs.get("json") or {})
        return FakeResponse()


def _channel() -> WhatsAppChannel:
    async def handler(message: InboundMessage) -> None:  # pragma: no cover
        raise AssertionError("no turn should run here")

    async def on_command(
        chat_id: str, session_id: str, message_id: str, _invocation
    ) -> None:  # pragma: no cover
        raise AssertionError("no command should run here")

    config = Config.load()
    object.__setattr__(config, "allowed_senders", frozenset({"212600000000"}))
    object.__setattr__(config, "text_batch_delay_seconds", 30.0)
    return WhatsAppChannel(config, handler, on_command)


class SendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.channel = _channel()
        self.http = FakeHttp()
        self.channel._http = self.http

    async def test_an_answer_quotes_the_message_it_answers(self):
        self.assertTrue(await self.channel.send("chat", "Voila.", "m7"))
        self.assertEqual(self.http.posts[0]["replyTo"], "m7")

    async def test_an_unattached_message_quotes_nothing(self):
        await self.channel.send("chat", "Voila.")
        self.assertNotIn("replyTo", self.http.posts[0])

    async def test_a_missing_id_is_not_sent_as_a_quote(self):
        """The bridge does not always give a message an id."""
        await self.channel.send("chat", "Voila.", "")
        self.assertNotIn("replyTo", self.http.posts[0])


class BatchTests(unittest.IsolatedAsyncioTestCase):
    """Three messages are one question, and the answer lands under the last."""

    def setUp(self):
        self.channel = _channel()

    def _event(self, text: str, message_id: str) -> Dict[str, Any]:
        return {
            "chatId": "chat",
            "messageId": message_id,
            "senderId": "212600000000@s.whatsapp.net",
            "senderNumber": "212600000000",
            "body": text,
        }

    async def test_the_batch_keeps_every_id_in_arrival_order(self):
        for index, part in enumerate(["about yesterday", "the invoice", "was it paid?"], 1):
            self.channel._accept(self._event(part, f"m{index}"))
        await asyncio.sleep(0)
        pending = self.channel._pending["chat"]
        self.assertEqual(pending.message_ids, ["m1", "m2", "m3"])
        # main.py quotes the last of them.
        self.assertEqual(pending.message_ids[-1], "m3")

    def tearDown(self):
        for task in list(self.channel._pending_tasks.values()):
            task.cancel()



class DirectMessageIdentityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.channel = _channel()

    def _event(
        self,
        chat_id: str,
        message_id: str,
        *,
        sender_number: str,
        identities: List[str],
    ) -> Dict[str, Any]:
        return {
            "chatId": chat_id,
            "messageId": message_id,
            "senderId": chat_id,
            "senderNumber": sender_number,
            "identities": identities,
            "body": "question",
        }

    async def test_phone_and_lid_dm_aliases_share_one_session(self):
        self.channel._accept(
            self._event(
                "999999999999999@lid",
                "m1",
                sender_number="999999999999999",
                identities=["999999999999999", "212600000000"],
            )
        )
        self.channel._accept(
            self._event(
                "212600000000@s.whatsapp.net",
                "m2",
                sender_number="212600000000",
                identities=["212600000000"],
            )
        )

        self.assertEqual(list(self.channel._pending), ["212600000000"])
        pending = self.channel._pending["212600000000"]
        self.assertEqual(pending.message_ids, ["m1", "m2"])
        self.assertEqual(pending.chat_id, "212600000000@s.whatsapp.net")

    def test_plus_prefixed_full_jid_allowlist_entry_matches(self):
        object.__setattr__(
            self.channel._config,
            "allowed_senders",
            frozenset({"+212600000000@s.whatsapp.net"}),
        )
        self.assertTrue(
            self.channel._is_allowed("212600000000@s.whatsapp.net", "", [])
        )

    def tearDown(self):
        for task in list(self.channel._pending_tasks.values()):
            task.cancel()

class GroupIsolationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.channel = _channel()
        object.__setattr__(self.channel._config, "answer_groups", True)
        object.__setattr__(
            self.channel._config,
            "allowed_senders",
            frozenset({"212600000000", "212611111111"}),
        )

    def _event(
        self,
        participant: str,
        message_id: str,
        *,
        sender_number: str = "",
        identities: List[str] | None = None,
    ) -> Dict[str, Any]:
        return {
            "chatId": "120363000000000000@g.us",
            "messageId": message_id,
            "senderId": participant,
            "senderNumber": sender_number,
            "identities": identities or [],
            "isGroup": True,
            "body": "question",
        }

    async def test_group_participants_never_share_a_batch_or_session(self):
        self.channel._accept(
            self._event("212600000000@s.whatsapp.net", "m1", sender_number="212600000000")
        )
        self.channel._accept(
            self._event("212611111111@s.whatsapp.net", "m2", sender_number="212611111111")
        )

        self.assertEqual(len(self.channel._pending), 2)
        self.assertEqual(
            {message.session_id for message in self.channel._pending.values()},
            {
                "120363000000000000@g.us:212600000000",
                "120363000000000000@g.us:212611111111",
            },
        )

    async def test_phone_and_lid_aliases_share_one_participant_session(self):
        self.channel._accept(
            self._event(
                "999999999999999@lid",
                "m1",
                sender_number="999999999999999",
                identities=["999999999999999", "212600000000"],
            )
        )
        self.channel._accept(
            self._event("212600000000@s.whatsapp.net", "m2", sender_number="212600000000")
        )

        self.assertEqual(len(self.channel._pending), 1)
        pending = next(iter(self.channel._pending.values()))
        self.assertEqual(pending.message_ids, ["m1", "m2"])
        self.assertEqual(pending.session_id, "120363000000000000@g.us:212600000000")

    def tearDown(self):
        for task in list(self.channel._pending_tasks.values()):
            task.cancel()


if __name__ == "__main__":
    unittest.main()
