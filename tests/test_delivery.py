from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from pilotage.delivery import (
    DeliveryStore,
    RECOVERED_MARKER,
    SendResult,
    claim_deliveries,
    compute_obligation_id,
    deliver_final,
    redeliver_claimed_deliveries,
    recover_deliveries,
)


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "delivery.db"
        self.store = DeliveryStore(self.path)

    def state(self, obligation_id: str) -> str:
        with closing(sqlite3.connect(self.path)) as connection:
            return str(
                connection.execute(
                    "SELECT state FROM delivery_obligations"
                    " WHERE obligation_id = ?",
                    (obligation_id,),
                ).fetchone()[0]
            )

    async def test_obligation_exists_before_send_and_finishes_delivered(self):
        content = "durable answer"
        obligation_id = compute_obligation_id("session", "message", content)
        seen = []

        async def send():
            seen.append(self.state(obligation_id))
            return SendResult(True)

        result = await deliver_final(
            self.store,
            session_key="session",
            message_ref="message",
            platform="whatsapp",
            chat_id="chat",
            thread_id="",
            content=content,
            send=send,
        )

        self.assertTrue(result)
        self.assertEqual(seen, ["attempting"])
        self.assertEqual(self.state(obligation_id), "delivered")

    async def test_retryable_send_is_retried_and_resolved(self):
        attempts = [
            SendResult(False, "offline", retryable=True),
            SendResult(True),
        ]

        async def send():
            return attempts.pop(0)

        with (
            mock.patch(
                "pilotage.delivery.asyncio.sleep",
                new=mock.AsyncMock(),
            ) as sleep,
            mock.patch("pilotage.delivery.random.uniform", return_value=0),
        ):
            result = await deliver_final(
                self.store,
                session_key="session",
                message_ref="message",
                platform="telegram",
                chat_id="42",
                thread_id="9",
                content="answer",
                send=send,
            )

        self.assertTrue(result)
        sleep.assert_awaited_once_with(2.0)

    async def test_failed_send_is_recovered_with_visible_duplicate_marker(self):
        content = "answer from before restart"

        async def reject():
            return SendResult(False, "rejected")

        await deliver_final(
            self.store,
            session_key="session",
            message_ref="message",
            platform="telegram",
            chat_id="42",
            thread_id="9",
            content=content,
            send=reject,
        )

        sent = []

        class Channel:
            async def send(self, *args, **kwargs):
                sent.append((args, kwargs))
                return SendResult(True)

        restarted = DeliveryStore(self.path)
        recovered = await recover_deliveries(
            restarted,
            {"telegram": Channel()},
        )

        self.assertEqual(recovered, 1)
        self.assertEqual(
            sent,
            [
                (
                    ("42", RECOVERED_MARKER + content),
                    {"thread_id": "9"},
                )
            ],
        )

    async def test_claimed_deliveries_wait_for_explicit_redelivery(self):
        content = "answer from before restart"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="9",
            content=content,
        )
        self.store.mark_failed(obligation_id, "rejected")
        sent = []

        class Channel:
            async def send(self, *args, **kwargs):
                sent.append((args, kwargs))
                return SendResult(True)

        restarted = DeliveryStore(self.path)
        rows = await claim_deliveries(restarted, {"telegram"})

        self.assertEqual(sent, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["obligation_id"], obligation_id)
        self.assertTrue(rows[0]["needs_marker"])

        recovered = await redeliver_claimed_deliveries(
            restarted,
            {"telegram": Channel()},
            rows,
        )

        self.assertEqual(recovered, 1)
        self.assertEqual(
            sent,
            [
                (
                    ("42", RECOVERED_MARKER + content),
                    {"thread_id": "9"},
                )
            ],
        )

    async def test_never_attempted_send_recovers_without_duplicate_marker(self):
        content = "not sent yet"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="whatsapp",
            chat_id="chat",
            thread_id="",
            content=content,
        )
        sent = []

        class Channel:
            async def send(self, chat_id, text):
                sent.append((chat_id, text))
                return True

        restarted = DeliveryStore(self.path)
        self.assertEqual(
            await recover_deliveries(
                restarted,
                {"whatsapp": Channel()},
            ),
            1,
        )
        self.assertEqual(sent, [("chat", content)])


if __name__ == "__main__":
    unittest.main()
