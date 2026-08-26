from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from pilotage.delivery import (
    DeliveryStore,
    DeliveryUnitLedger,
    MAX_ATTEMPTS,
    RECOVERED_MARKER,
    SendResult,
    STALE_AFTER_SECONDS,
    claim_deliveries,
    claim_live_deliveries,
    compute_obligation_id,
    delivery_fingerprint,
    deliver_final,
    redeliver_claimed_deliveries,
    recover_deliveries,
    recover_live_deliveries,
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

    def row(self, obligation_id: str):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            return connection.execute(
                "SELECT * FROM delivery_obligations WHERE obligation_id = ?",
                (obligation_id,),
            ).fetchone()

    def unit_rows(self, obligation_id: str):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            return connection.execute(
                "SELECT * FROM delivery_units WHERE obligation_id = ?"
                " ORDER BY position",
                (obligation_id,),
            ).fetchall()

    def test_readiness_probe_writes_the_real_database_without_leaving_a_row(self):
        self.store.verify_writable()

        with closing(sqlite3.connect(self.path)) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            obligations = connection.execute(
                "SELECT COUNT(*) FROM delivery_obligations"
            ).fetchone()[0]

        self.assertIn("delivery_obligations", tables)
        self.assertIn("delivery_units", tables)
        self.assertEqual(obligations, 0)

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

    async def test_storage_failure_prevents_an_unledgered_send(self):
        send = mock.AsyncMock(return_value=SendResult(True))
        with mock.patch.object(
            self.store,
            "record",
            side_effect=sqlite3.OperationalError("disk unavailable"),
        ):
            result = await deliver_final(
                self.store,
                session_key="session",
                message_ref="message",
                platform="whatsapp",
                chat_id="chat",
                thread_id="",
                content="must remain unsent",
                send=send,
            )

        self.assertFalse(result)
        self.assertIn("not durably claimed", result.error)
        send.assert_not_awaited()

    async def test_unclaimed_obligation_prevents_a_send(self):
        send = mock.AsyncMock(return_value=SendResult(True))
        with mock.patch.object(self.store, "mark_attempting", return_value=False):
            result = await deliver_final(
                self.store,
                session_key="session",
                message_ref="message",
                platform="telegram",
                chat_id="42",
                thread_id="",
                content="must remain pending",
                send=send,
            )

        self.assertFalse(result)
        send.assert_not_awaited()
        obligation_id = compute_obligation_id(
            "session", "message", "must remain pending"
        )
        self.assertEqual(self.state(obligation_id), "pending")

    async def test_failed_settlement_is_reported_as_non_durable(self):
        async def reject():
            return SendResult(False, "offline", retryable=True)

        with (
            mock.patch.object(self.store, "mark_failed", return_value=False),
            mock.patch("pilotage.delivery.asyncio.sleep", new=mock.AsyncMock()),
        ):
            result = await deliver_final(
                self.store,
                session_key="session",
                message_ref="message",
                platform="telegram",
                chat_id="42",
                thread_id="",
                content="answer",
                send=reject,
            )

        self.assertFalse(result)
        self.assertIn("not durably recorded", result.error)

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

    async def test_unitized_retry_sends_only_the_missing_unit(self):
        content = "answer with a file"
        obligation_id = compute_obligation_id("session", "message", content)
        sent = []
        file_attempts = [
            SendResult(False, "offline", retryable=True),
            SendResult(True, message_id="file-message"),
        ]

        async def legacy_send():
            raise AssertionError("unitized delivery must use the ledger sender")

        async def ledger_send(ledger: DeliveryUnitLedger):
            units = await ledger.prepare(
                [
                    ("text", delivery_fingerprint("text", "answer")),
                    ("file", delivery_fingerprint("file", "sha256")),
                ]
            )

            async def send_text():
                sent.append("text")
                return SendResult(True, message_id="text-message")

            async def send_file():
                sent.append("file")
                return file_attempts.pop(0)

            result = await ledger.run(units[0], send_text)
            if not result:
                return result
            return await ledger.run(units[1], send_file)

        with (
            mock.patch("pilotage.delivery.asyncio.sleep", new=mock.AsyncMock()),
            mock.patch("pilotage.delivery.random.uniform", return_value=0),
        ):
            result = await deliver_final(
                self.store,
                session_key="session",
                message_ref="message",
                platform="telegram",
                chat_id="42",
                thread_id="",
                content=content,
                send=legacy_send,
                ledger_send=ledger_send,
            )

        self.assertTrue(result)
        self.assertEqual(sent, ["text", "file", "file"])
        self.assertEqual(self.state(obligation_id), "delivered")
        rows = self.unit_rows(obligation_id)
        self.assertEqual([row["state"] for row in rows], ["delivered", "delivered"])
        self.assertEqual(rows[0]["evidence"], "text-message")
        self.assertEqual(rows[1]["evidence"], "file-message")

    async def test_unitized_crash_recovers_only_pending_units(self):
        content = "answer with a file"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="whatsapp",
            chat_id="chat",
            thread_id="",
            content=content,
        )
        self.assertTrue(self.store.mark_attempting(obligation_id))
        ledger = DeliveryUnitLedger(self.store, obligation_id)
        descriptors = [
            ("text", delivery_fingerprint("text", "answer")),
            ("file", delivery_fingerprint("file", "sha256")),
        ]
        units = await ledger.prepare(descriptors)

        async def send_text():
            return SendResult(True, message_id="text-message")

        self.assertTrue(
            await ledger.run(units[0], send_text)
        )

        restarted = DeliveryStore(self.path)
        rows = await claim_deliveries(restarted, {"whatsapp"})
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["unitized"])
        self.assertFalse(rows[0]["needs_marker"])
        sent = []
        owner = self

        class Channel:
            async def send(self, _chat_id, received, *, delivery_ledger):
                owner.assertEqual(received, content)
                retry_units = await delivery_ledger.prepare(descriptors)

                async def duplicate_text():
                    raise AssertionError("delivered text must be skipped")

                async def send_file():
                    sent.append("file")
                    return SendResult(True, message_id="file-message")

                result = await delivery_ledger.run(retry_units[0], duplicate_text)
                if not result:
                    return result
                return await delivery_ledger.run(retry_units[1], send_file)

        recovered = await redeliver_claimed_deliveries(
            restarted,
            {"whatsapp": Channel()},
            rows,
        )

        self.assertEqual(recovered, 1)
        self.assertEqual(sent, ["file"])
        self.assertEqual(self.state(obligation_id), "delivered")

    async def test_unitized_crash_during_a_unit_still_fails_closed(self):
        content = "answer with a file"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content=content,
        )
        self.assertTrue(self.store.mark_attempting(obligation_id))
        ledger = DeliveryUnitLedger(self.store, obligation_id)
        units = await ledger.prepare(
            [("file", delivery_fingerprint("file", "sha256"))]
        )
        self.assertTrue(
            self.store.mark_unit_attempting(obligation_id, units[0].unit_id)
        )

        self.assertEqual(
            await claim_deliveries(DeliveryStore(self.path), {"telegram"}),
            [],
        )

    async def test_failed_send_is_recovered_with_visible_duplicate_marker(self):
        content = "answer from before restart"

        async def reject():
            return SendResult(False, "rejected", retryable=True)

        with mock.patch(
            "pilotage.delivery.asyncio.sleep", new=mock.AsyncMock()
        ):
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
        self.store.mark_attempting(obligation_id)
        self.store.mark_failed(
            obligation_id,
            "rejected",
            retry_safe=True,
        )
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

    async def test_recovery_preserves_the_original_reply_target(self):
        content = "quoted answer"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="9",
            reply_to="777",
            content=content,
        )
        self.store.mark_attempting(obligation_id)
        self.store.mark_failed(obligation_id, "offline", retry_safe=True)
        sent = []

        class Channel:
            async def send(self, *args, **kwargs):
                sent.append((args, kwargs))
                return SendResult(True)

        recovered = await recover_deliveries(
            DeliveryStore(self.path),
            {"telegram": Channel()},
        )

        self.assertEqual(recovered, 1)
        self.assertEqual(
            sent,
            [
                (
                    ("42", RECOVERED_MARKER + content, "777"),
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

    async def test_live_owner_retries_safe_failure_after_freshness_floor(self):
        content = "retry me while running"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="whatsapp",
            chat_id="chat",
            thread_id="",
            content=content,
        )
        self.store.mark_attempting(obligation_id)
        self.store.mark_failed(
            obligation_id,
            "503",
            retry_safe=True,
        )
        failed_at = float(self.row(obligation_id)["updated_at"])
        sent = []

        class Channel:
            async def send(self, chat_id, text):
                sent.append((chat_id, text))
                return SendResult(True)

        self.assertEqual(
            await recover_live_deliveries(
                self.store,
                {"whatsapp": Channel()},
                now=failed_at + 59,
            ),
            0,
        )
        self.assertEqual(
            await recover_live_deliveries(
                self.store,
                {"whatsapp": Channel()},
                now=failed_at + 61,
            ),
            1,
        )
        self.assertEqual(
            sent,
            [("chat", RECOVERED_MARKER + content)],
        )

    async def test_live_claim_is_fenced_against_a_second_sweep(self):
        obligation_id = compute_obligation_id("session", "message", "answer")
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content="answer",
        )
        self.store.mark_attempting(obligation_id)
        self.store.mark_failed(obligation_id, "503", retry_safe=True)
        failed_at = float(self.row(obligation_id)["updated_at"])

        first = await claim_live_deliveries(
            self.store,
            {"telegram"},
            now=failed_at + 61,
        )
        second = await claim_live_deliveries(
            self.store,
            {"telegram"},
            now=failed_at + 122,
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        self.assertEqual(self.state(obligation_id), "attempting")

    async def test_unknown_acceptance_is_never_retried(self):
        obligation_id = compute_obligation_id("session", "message", "answer")
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content="answer",
        )
        self.store.mark_attempting(obligation_id)
        self.store.mark_failed(obligation_id, "response timed out")
        failed_at = float(self.row(obligation_id)["updated_at"])

        self.assertEqual(
            await claim_live_deliveries(
                self.store,
                {"telegram"},
                now=failed_at + 3600,
            ),
            [],
        )
        self.assertEqual(
            await claim_deliveries(DeliveryStore(self.path), {"telegram"}),
            [],
        )

    async def test_crash_during_send_is_not_retried_as_known_rejection(self):
        obligation_id = compute_obligation_id("session", "message", "answer")
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content="answer",
        )
        self.store.mark_attempting(obligation_id)

        self.assertEqual(
            await claim_deliveries(DeliveryStore(self.path), {"telegram"}),
            [],
        )
        self.assertEqual(self.state(obligation_id), "attempting")

    async def test_old_owner_cannot_settle_a_reclaimed_row(self):
        obligation_id = compute_obligation_id("session", "message", "answer")
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content="answer",
        )
        self.store.mark_attempting(obligation_id)
        self.store.mark_failed(obligation_id, "503", retry_safe=True)

        restarted = DeliveryStore(self.path)
        claimed = await claim_deliveries(restarted, {"telegram"})

        self.assertEqual(len(claimed), 1)
        self.assertFalse(self.store.mark_delivered(obligation_id))
        self.assertEqual(self.state(obligation_id), "attempting")

    async def test_live_retry_attempt_cap_abandons_the_row(self):
        obligation_id = compute_obligation_id("session", "message", "answer")
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content="answer",
        )
        self.store.mark_attempting(obligation_id)
        self.store.mark_failed(obligation_id, "503", retry_safe=True)
        failed_at = float(self.row(obligation_id)["updated_at"])
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "UPDATE delivery_obligations SET attempts = ?"
                " WHERE obligation_id = ?",
                (MAX_ATTEMPTS, obligation_id),
            )
            connection.commit()

        self.assertEqual(
            await claim_live_deliveries(
                self.store,
                {"telegram"},
                now=failed_at + 61,
            ),
            [],
        )
        self.assertEqual(self.state(obligation_id), "abandoned")

    async def test_stale_live_retry_is_abandoned_without_sending(self):
        obligation_id = compute_obligation_id("session", "message", "answer")
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content="answer",
        )
        self.store.mark_attempting(obligation_id)
        self.store.mark_failed(obligation_id, "503", retry_safe=True)
        row = self.row(obligation_id)

        self.assertEqual(
            await claim_live_deliveries(
                self.store,
                {"telegram"},
                now=float(row["created_at"]) + STALE_AFTER_SECONDS + 1,
                min_age_seconds=0,
            ),
            [],
        )
        self.assertEqual(self.state(obligation_id), "abandoned")


if __name__ == "__main__":
    unittest.main()
