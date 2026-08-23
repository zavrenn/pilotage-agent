"""Hermes-derived WhatsApp group admission and mention behavior."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest import mock

from pilotage.channels.whatsapp import InboundMessage, WhatsAppChannel
from pilotage.config import Config


async def _handle(_message: InboundMessage) -> None:  # pragma: no cover
    raise AssertionError("no turn should run")


async def _command(*_args) -> None:  # pragma: no cover
    raise AssertionError("no command should run")


class GroupAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        environment = mock.patch.dict(
            os.environ,
            {
                "PILOTAGE_HOME": str(root),
                "PILOTAGE_CONFIG": "",
                "PILOTAGE_ALLOWED_SENDERS": "",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        config = Config.load()
        object.__setattr__(config, "group_policy", "allowlist")
        object.__setattr__(
            config,
            "group_allow_from",
            frozenset({"120363001234567890@g.us"}),
        )
        object.__setattr__(config, "require_mention", True)
        object.__setattr__(config, "text_batch_delay_seconds", 30.0)
        self.channel = WhatsAppChannel(config, _handle, _command)

    def _event(self, body: str = "hello", **overrides: Any) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "isGroup": True,
            "body": body,
            "chatId": "120363001234567890@g.us",
            "messageId": "m1",
            "senderId": "212600000000@s.whatsapp.net",
            "senderNumber": "212600000000",
            "mentionedIds": [],
            "botIds": [
                "15551230000@10@s.whatsapp.net",
                "67427329167522@lid",
            ],
            "quotedParticipant": "",
        }
        event.update(overrides)
        return event

    async def test_unmentioned_group_message_is_ignored(self):
        self.channel._accept(self._event())
        self.assertEqual(self.channel._pending, {})

    async def test_direct_mention_is_accepted_without_dm_allowlist(self):
        self.channel._accept(
            self._event(
                "@15551230000 what is the weather?",
                mentionedIds=["15551230000@s.whatsapp.net"],
            )
        )

        pending = next(iter(self.channel._pending.values()))
        self.assertEqual(pending.text, "what is the weather?")
        self.assertEqual(
            pending.session_id,
            "120363001234567890@g.us:212600000000",
        )

    async def test_mention_never_bypasses_the_group_allowlist(self):
        self.channel._accept(
            self._event(
                "@15551230000 hello",
                chatId="120363999999999999@g.us",
                mentionedIds=["15551230000@s.whatsapp.net"],
            )
        )
        self.assertEqual(self.channel._pending, {})

    def test_reply_to_either_bot_identity_is_a_trigger(self):
        self.assertTrue(
            self.channel._group_message_is_triggered(
                self._event(
                    "replying",
                    quotedParticipant="67427329167522:4@lid",
                )
            )
        )

    def test_group_command_is_a_trigger(self):
        self.assertTrue(
            self.channel._group_message_is_triggered(self._event("/status"))
        )

    def test_another_users_mention_is_not_a_trigger(self):
        self.assertFalse(
            self.channel._group_message_is_triggered(
                self._event(
                    "@19990000000 hello",
                    mentionedIds=["19990000000@s.whatsapp.net"],
                )
            )
        )

    def tearDown(self):
        for task in list(self.channel._pending_tasks.values()):
            task.cancel()


if __name__ == "__main__":
    unittest.main()
