from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from pilotage.delivery import (
    DeliveryPlanError,
    DeliveryStore,
    DeliveryUnit,
    DeliveryUnitLedger,
    MAX_ATTEMPTS,
    MAX_ROWS,
    RETENTION_SECONDS,
    SendResult,
    STALE_AFTER_SECONDS,
    claim_deliveries,
    claim_live_deliveries,
    compute_command_id,
    compute_obligation_id,
    delivery_fingerprint,
    deliver_final,
    redeliver_claimed_deliveries,
    recover_deliveries,
    recover_live_deliveries,
    send_with_retry,
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
        self.assertIn("command_outcomes", tables)
        self.assertEqual(obligations, 0)

    def test_command_outcome_executes_once_and_replays_exact_response(self):
        command_id = compute_command_id("whatsapp", "claim-1")
        arguments = {
            "command_id": command_id,
            "platform": "whatsapp",
            "claim_id": "claim-1",
            "session_key": "session",
            "command_name": "new",
            "arguments": "",
        }

        first = self.store.begin_command(**arguments)
        self.assertTrue(first.execute)
        self.assertFalse(first.completed)

        self.store.complete_command(command_id, "Starting fresh")
        replay = self.store.begin_command(**arguments)

        self.assertFalse(replay.execute)
        self.assertTrue(replay.completed)
        self.assertEqual(replay.response, "Starting fresh")

    def test_old_completed_command_survives_replay_pruning(self):
        command_id = compute_command_id("telegram", "claim-old")
        arguments = {
            "command_id": command_id,
            "platform": "telegram",
            "claim_id": "claim-old",
            "session_key": "session",
            "command_name": "approve",
            "arguments": "",
        }
        self.store.begin_command(**arguments)
        self.store.complete_command(command_id, "Approved")
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "UPDATE command_outcomes SET updated_at = 0"
                " WHERE command_id = ?",
                (command_id,),
            )
            connection.commit()

        first_replay = self.store.begin_command(**arguments)
        restarted = DeliveryStore(self.path)
        second_replay = restarted.begin_command(**arguments)

        self.assertFalse(first_replay.execute)
        self.assertTrue(first_replay.completed)
        self.assertEqual(first_replay.response, "Approved")
        self.assertFalse(second_replay.execute)
        self.assertTrue(second_replay.completed)
        self.assertEqual(second_replay.response, "Approved")

    def test_interrupted_command_is_not_executed_again(self):
        command_id = compute_command_id("telegram", "claim-2")
        arguments = {
            "command_id": command_id,
            "platform": "telegram",
            "claim_id": "claim-2",
            "session_key": "session",
            "command_name": "approve",
            "arguments": "",
        }
        self.assertTrue(self.store.begin_command(**arguments).execute)

        interrupted = self.store.begin_command(**arguments)

        self.assertFalse(interrupted.execute)
        self.assertFalse(interrupted.completed)
        self.store.complete_command(command_id, "Outcome unknown")
        self.assertEqual(
            self.store.begin_command(**arguments).response,
            "Outcome unknown",
        )

    def test_command_identity_cannot_change_on_replay(self):
        command_id = compute_command_id("telegram", "claim-3")
        self.store.begin_command(
            command_id=command_id,
            platform="telegram",
            claim_id="claim-3",
            session_key="session",
            command_name="deny",
            arguments="reason",
        )

        with self.assertRaisesRegex(DeliveryPlanError, "collides"):
            self.store.begin_command(
                command_id=command_id,
                platform="telegram",
                claim_id="claim-3",
                session_key="session",
                command_name="approve",
                arguments="",
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

    def test_obligation_identity_is_immutable(self):
        obligation_id = "fixed-identity"
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="7",
            reply_to="99",
            content="original",
        )

        with self.assertRaisesRegex(DeliveryPlanError, "collides"):
            self.store.record(
                obligation_id=obligation_id,
                session_key="session",
                platform="telegram",
                chat_id="42",
                thread_id="7",
                reply_to="99",
                content="replacement",
            )

        self.assertEqual(self.row(obligation_id)["content"], "original")

    async def test_a_delivered_obligation_is_never_reset_and_resent(self):
        first_send = mock.AsyncMock(return_value=SendResult(True))
        arguments = {
            "session_key": "session",
            "message_ref": "message",
            "platform": "whatsapp",
            "chat_id": "chat",
            "thread_id": "",
            "content": "exact answer",
        }
        self.assertTrue(
            await deliver_final(self.store, send=first_send, **arguments)
        )

        duplicate_send = mock.AsyncMock(return_value=SendResult(True))
        duplicate = await deliver_final(
            self.store,
            send=duplicate_send,
            **arguments,
        )

        self.assertFalse(duplicate)
        duplicate_send.assert_not_awaited()
        obligation_id = compute_obligation_id(
            "session", "message", "exact answer"
        )
        self.assertEqual(self.state(obligation_id), "delivered")

    def test_capacity_never_prunes_an_unresolved_obligation(self):
        for index in range(MAX_ROWS):
            self.store.record(
                obligation_id=f"pending-{index}",
                session_key=f"session-{index}",
                platform="telegram",
                chat_id="42",
                thread_id="",
                reply_to=str(index),
                content="answer",
            )

        with self.assertRaisesRegex(DeliveryPlanError, "capacity"):
            self.store.record(
                obligation_id="one-too-many",
                session_key="new-session",
                platform="telegram",
                chat_id="42",
                thread_id="",
                reply_to="new-message",
                content="new answer",
            )

        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                "SELECT obligation_id, state FROM delivery_obligations"
            ).fetchall()
        self.assertEqual(len(rows), MAX_ROWS)
        self.assertTrue(all(state == "pending" for _, state in rows))
        self.assertNotIn("one-too-many", {identity for identity, _ in rows})

    def test_stale_failures_release_capacity_without_dropping_their_fences(self):
        self.store.verify_writable()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executemany(
                "INSERT INTO delivery_obligations"
                " (obligation_id, session_key, platform, chat_id, thread_id,"
                " reply_to, content, state, attempts, created_at, updated_at,"
                " owner_token, last_error, retry_safe, next_attempt_at)"
                " VALUES (?, ?, 'telegram', '42', NULL, NULL, 'uncertain',"
                " 'failed', 1, 0, 0, 'old-owner', 'timed out', 0, 0)",
                [
                    (f"uncertain-{index}", f"session-{index}")
                    for index in range(MAX_ROWS)
                ],
            )
            connection.commit()

        self.store.record(
            obligation_id="new-pending",
            session_key="new-session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content="new answer",
        )

        with closing(sqlite3.connect(self.path)) as connection:
            counts = dict(
                connection.execute(
                    "SELECT state, COUNT(*) FROM delivery_obligations"
                    " GROUP BY state"
                ).fetchall()
            )
        self.assertEqual(counts, {"abandoned": MAX_ROWS, "pending": 1})

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
        with mock.patch.object(
            self.store,
            "mark_attempting",
            side_effect=sqlite3.OperationalError("claim write interrupted"),
        ):
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

    async def test_current_owner_pending_obligation_is_recovered_live(self):
        content = "must not wait for a restart"
        send = mock.AsyncMock(return_value=SendResult(True))
        with mock.patch.object(self.store, "mark_attempting", return_value=False):
            result = await deliver_final(
                self.store,
                session_key="session",
                message_ref="message",
                platform="telegram",
                chat_id="42",
                thread_id="",
                content=content,
                send=send,
            )
        self.assertFalse(result)
        obligation_id = compute_obligation_id("session", "message", content)
        pending_at = float(self.row(obligation_id)["updated_at"])

        class Channel:
            async def send(self, _chat_id, text, *, delivery_ledger, **_kwargs):
                units = await delivery_ledger.prepare(
                    [("text", delivery_fingerprint("telegram-text", text))]
                )

                async def accepted():
                    return SendResult(True)

                return await delivery_ledger.run(units[0], accepted)

        recovered = await recover_live_deliveries(
            self.store,
            {"telegram": Channel()},
            now=pending_at + 61,
        )

        self.assertEqual(recovered, 1)
        self.assertEqual(self.state(obligation_id), "delivered")

    async def test_unprepared_ledger_callback_is_quarantined_without_retry(self):
        content = "must use the exact plan"
        obligation_id = compute_obligation_id("session", "message", content)
        calls = []

        async def bypassed_plan(_ledger):
            calls.append("callback")
            return SendResult(True)

        result = await deliver_final(
            self.store,
            session_key="session",
            message_ref="message",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content=content,
            send=mock.AsyncMock(),
            ledger_send=bypassed_plan,
        )

        self.assertFalse(result)
        self.assertEqual(calls, ["callback"])
        self.assertEqual(self.state(obligation_id), "failed")
        self.assertFalse(bool(self.row(obligation_id)["retry_safe"]))
        self.assertEqual(
            await claim_deliveries(DeliveryStore(self.path), {"telegram"}),
            [],
        )

    async def test_concurrent_ledger_loser_cannot_clobber_inflight_winner(self):
        content = "one exact concurrent reply"
        obligation_id = compute_obligation_id("session", "message", content)
        descriptors = [
            ("text", delivery_fingerprint("telegram-text", content))
        ]
        winner_prepared = asyncio.Event()
        loser_prepared = asyncio.Event()
        winner_network = asyncio.Event()
        release_winner = asyncio.Event()
        loser_network_called = []

        async def winner_send(ledger):
            units = await ledger.prepare(descriptors)
            winner_prepared.set()
            await loser_prepared.wait()

            async def accepted():
                winner_network.set()
                await release_winner.wait()
                return SendResult(True, message_id="winner")

            return await ledger.run(units[0], accepted)

        async def loser_send(ledger):
            units = await ledger.prepare(descriptors)
            loser_prepared.set()
            await winner_network.wait()

            async def must_not_send():
                loser_network_called.append(True)
                return SendResult(True, message_id="loser")

            return await ledger.run(units[0], must_not_send)

        arguments = {
            "session_key": "session",
            "message_ref": "message",
            "platform": "telegram",
            "chat_id": "42",
            "thread_id": "",
            "content": content,
            "send": mock.AsyncMock(),
        }
        winner = asyncio.create_task(
            deliver_final(self.store, ledger_send=winner_send, **arguments)
        )
        await winner_prepared.wait()
        loser = asyncio.create_task(
            deliver_final(self.store, ledger_send=loser_send, **arguments)
        )

        loser_result = await asyncio.wait_for(loser, 1.0)
        self.assertFalse(loser_result)
        self.assertEqual(loser_network_called, [])
        self.assertEqual(self.state(obligation_id), "attempting")
        self.assertEqual(self.unit_rows(obligation_id)[0]["state"], "attempting")

        release_winner.set()
        self.assertTrue(await asyncio.wait_for(winner, 1.0))
        self.assertEqual(self.state(obligation_id), "delivered")

    async def test_concurrent_loser_cannot_downgrade_winners_safe_failure(self):
        content = "one safely rejected concurrent reply"
        obligation_id = compute_obligation_id("session", "message", content)
        descriptors = [
            ("text", delivery_fingerprint("telegram-text", content))
        ]
        winner_prepared = asyncio.Event()
        loser_prepared = asyncio.Event()
        winner_network = asyncio.Event()
        loser_lost_claim = asyncio.Event()
        winner_unit_failed = asyncio.Event()
        release_winner_result = asyncio.Event()
        loser_network = mock.AsyncMock(return_value=SendResult(True))

        async def winner_send(ledger):
            units = await ledger.prepare(descriptors)
            winner_prepared.set()
            await loser_prepared.wait()

            async def safely_rejected():
                winner_network.set()
                await loser_lost_claim.wait()
                return SendResult(
                    False,
                    "service unavailable",
                    retryable=True,
                    retry_after=30,
                )

            result = await ledger.run(units[0], safely_rejected)
            winner_unit_failed.set()
            await release_winner_result.wait()
            return result

        async def loser_send(ledger):
            units = await ledger.prepare(descriptors)
            loser_prepared.set()
            await winner_network.wait()
            result = await ledger.run(units[0], loser_network)
            loser_lost_claim.set()
            await winner_unit_failed.wait()
            return result

        arguments = {
            "session_key": "session",
            "message_ref": "message",
            "platform": "telegram",
            "chat_id": "42",
            "thread_id": "",
            "content": content,
            "send": mock.AsyncMock(),
        }
        winner = asyncio.create_task(
            deliver_final(self.store, ledger_send=winner_send, **arguments)
        )
        await winner_prepared.wait()
        loser = asyncio.create_task(
            deliver_final(self.store, ledger_send=loser_send, **arguments)
        )

        loser_result = await asyncio.wait_for(loser, 1.0)
        self.assertFalse(loser_result)
        loser_network.assert_not_awaited()
        self.assertEqual(self.state(obligation_id), "attempting")
        unit = self.unit_rows(obligation_id)[0]
        self.assertEqual(unit["state"], "failed")
        self.assertTrue(bool(unit["retry_safe"]))

        release_winner_result.set()
        self.assertFalse(await asyncio.wait_for(winner, 1.0))
        parent = self.row(obligation_id)
        self.assertEqual(parent["state"], "failed")
        self.assertTrue(bool(parent["retry_safe"]))
        recovered = await claim_deliveries(
            DeliveryStore(self.path),
            {"telegram"},
            now=float(parent["next_attempt_at"]) + 1,
        )
        self.assertEqual(len(recovered), 1)

    async def test_plan_preparation_failure_keeps_inbound_fence_unowned(self):
        content = "planning must finish first"
        obligation_id = compute_obligation_id("session", "message", content)
        network = mock.AsyncMock(return_value=SendResult(True))

        async def planned_send(ledger):
            units = await ledger.prepare(
                [("text", delivery_fingerprint("telegram-text", content))]
            )
            return await ledger.run(units[0], network)

        with mock.patch.object(
            self.store,
            "record_units",
            side_effect=sqlite3.OperationalError("plan write failed"),
        ):
            result = await deliver_final(
                self.store,
                session_key="session",
                message_ref="message",
                platform="telegram",
                chat_id="42",
                thread_id="",
                content=content,
                send=mock.AsyncMock(),
                ledger_send=planned_send,
            )

        self.assertFalse(result)
        network.assert_not_awaited()
        self.assertIsNone(self.row(obligation_id))

    async def test_live_sweep_recovers_claim_db_failure_before_network(self):
        content = "recover a safe activated plan"
        obligation_id = compute_obligation_id("session", "message", content)
        network = mock.AsyncMock(return_value=SendResult(True))

        async def planned_send(ledger):
            units = await ledger.prepare(
                [("text", delivery_fingerprint("telegram-text", content))]
            )
            return await ledger.run(units[0], network)

        with mock.patch.object(
            self.store,
            "mark_unit_attempting",
            side_effect=sqlite3.OperationalError("claim write failed"),
        ):
            result = await deliver_final(
                self.store,
                session_key="session",
                message_ref="message",
                platform="telegram",
                chat_id="42",
                thread_id="",
                content=content,
                send=mock.AsyncMock(),
                ledger_send=planned_send,
            )

        self.assertFalse(result)
        network.assert_not_awaited()
        self.assertEqual(self.state(obligation_id), "attempting")
        self.assertEqual(self.unit_rows(obligation_id)[0]["state"], "pending")
        failed_at = float(self.row(obligation_id)["updated_at"])

        class Channel:
            async def send(self, _chat_id, text, *, delivery_ledger, **_kwargs):
                units = await delivery_ledger.prepare(
                    [("text", delivery_fingerprint("telegram-text", text))]
                )
                return await delivery_ledger.run(
                    units[0],
                    mock.AsyncMock(return_value=SendResult(True)),
                )

        recovered = await recover_live_deliveries(
            self.store,
            {"telegram": Channel()},
            now=failed_at + 61,
        )

        self.assertEqual(recovered, 1)
        self.assertEqual(self.state(obligation_id), "delivered")

    async def test_startup_claim_failure_is_not_treated_as_an_empty_ledger(self):
        with mock.patch.object(
            self.store,
            "claim_recoverable",
            side_effect=sqlite3.DatabaseError("database is malformed"),
        ):
            with self.assertRaisesRegex(sqlite3.DatabaseError, "malformed"):
                await claim_deliveries(self.store, {"telegram"})

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

    async def test_long_retry_after_is_deferred_without_blocking(self):
        send = mock.AsyncMock(
            return_value=SendResult(
                False,
                "flood limited",
                retryable=True,
                retry_after=5_827.0,
            )
        )
        with mock.patch(
            "pilotage.delivery.asyncio.sleep",
            new=mock.AsyncMock(),
        ) as sleep:
            result = await send_with_retry(send)

        self.assertFalse(result)
        self.assertEqual(result.retry_after, 5_827.0)
        send.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_retry_after_due_time_survives_in_durable_obligation(self):
        content = "defer this reply"
        obligation_id = compute_obligation_id("session", "message", content)
        now = 1_800_000_000.0
        send = mock.AsyncMock(
            return_value=SendResult(
                False,
                "flood limited",
                retryable=True,
                retry_after=120.0,
            )
        )
        with mock.patch("pilotage.delivery.time.time", return_value=now):
            result = await deliver_final(
                self.store,
                session_key="session",
                message_ref="message",
                platform="telegram",
                chat_id="42",
                thread_id="",
                content=content,
                send=send,
            )

        self.assertFalse(result)
        self.assertEqual(float(self.row(obligation_id)["next_attempt_at"]), now + 120)
        restarted = DeliveryStore(self.path)

        class Channel:
            async def send(self, _chat_id, text, *, delivery_ledger, **_kwargs):
                units = await delivery_ledger.prepare(
                    [("text", delivery_fingerprint("telegram-text", text))]
                )

                async def accepted():
                    return SendResult(True)

                return await delivery_ledger.run(units[0], accepted)

        self.assertEqual(
            await recover_live_deliveries(
                restarted,
                {"telegram": Channel()},
                now=now + 119,
                min_age_seconds=0,
            ),
            0,
        )
        self.assertEqual(
            await recover_live_deliveries(
                restarted,
                {"telegram": Channel()},
                now=now + 121,
                min_age_seconds=0,
            ),
            1,
        )

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

    async def test_crash_after_plan_before_activation_is_safely_recovered(self):
        content = "planned before crash"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content=content,
        )
        descriptors = [
            ("text", delivery_fingerprint("telegram-text", content))
        ]
        ledger = DeliveryUnitLedger(self.store, obligation_id)
        await ledger.prepare(descriptors)

        self.assertEqual(self.state(obligation_id), "pending")
        self.assertTrue(self.store.has_unit_plan(obligation_id))

        restarted = DeliveryStore(self.path)
        rows = await claim_deliveries(restarted, {"telegram"})
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["unitized"])
        self.assertEqual(self.state(obligation_id), "pending")
        sent = []

        class Channel:
            async def send(self, _chat_id, text, *, delivery_ledger, **_kwargs):
                units = await delivery_ledger.prepare(descriptors)

                async def accepted():
                    sent.append(text)
                    return SendResult(True, message_id="recovered")

                return await delivery_ledger.run(units[0], accepted)

        self.assertEqual(
            await redeliver_claimed_deliveries(
                restarted,
                {"telegram": Channel()},
                rows,
            ),
            1,
        )
        self.assertEqual(sent, [content])
        self.assertEqual(self.state(obligation_id), "delivered")

    async def test_unitized_recovery_refuses_an_adapter_without_ledger_support(self):
        content = "one unit already planned"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="whatsapp",
            chat_id="chat",
            thread_id="",
            content=content,
        )
        ledger = DeliveryUnitLedger(self.store, obligation_id)
        await ledger.prepare(
            [("text", delivery_fingerprint("whatsapp-text", content))]
        )
        restarted = DeliveryStore(self.path)
        rows = await claim_deliveries(restarted, {"whatsapp"})
        sends = []

        class LegacyChannel:
            async def send(self, chat_id, text):
                sends.append((chat_id, text))
                return SendResult(True)

        self.assertEqual(
            await redeliver_claimed_deliveries(
                restarted,
                {"whatsapp": LegacyChannel()},
                rows,
            ),
            0,
        )
        self.assertEqual(sends, [])
        self.assertEqual(self.state(obligation_id), "pending")

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

    async def test_unit_retry_after_is_durable_before_parent_settlement(self):
        content = "retry after a crash"
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
            [("text", delivery_fingerprint("text", content))]
        )
        failed_at = float(self.row(obligation_id)["updated_at"]) + 1.0

        async def flood_limited():
            return SendResult(
                False,
                "flood limited",
                retryable=True,
                retry_after=120.0,
            )

        with mock.patch("pilotage.delivery.time.time", return_value=failed_at):
            result = await ledger.run(units[0], flood_limited)

        self.assertFalse(result)
        self.assertEqual(self.state(obligation_id), "attempting")
        self.assertEqual(
            float(self.row(obligation_id)["next_attempt_at"]),
            failed_at + 120.0,
        )

        restarted = DeliveryStore(self.path)
        self.assertEqual(
            await claim_deliveries(
                restarted,
                {"telegram"},
                now=failed_at + 119.0,
            ),
            [],
        )
        claimed = await claim_deliveries(
            restarted,
            {"telegram"},
            now=failed_at + 121.0,
        )
        self.assertEqual(len(claimed), 1)
        self.assertTrue(claimed[0]["unitized"])
        self.assertFalse(claimed[0]["needs_marker"])

    async def test_proven_unsent_failure_recovers_without_duplicate_marker(self):
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
            async def send(self, *args, delivery_ledger, **kwargs):
                units = await delivery_ledger.prepare(
                    [("text", delivery_fingerprint("telegram-text", args[1]))]
                )

                async def accepted():
                    sent.append((args, kwargs))
                    return SendResult(True)

                return await delivery_ledger.run(units[0], accepted)

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
                    ("42", content),
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
            async def send(self, *args, delivery_ledger, **kwargs):
                units = await delivery_ledger.prepare(
                    [("text", delivery_fingerprint("telegram-text", args[1]))]
                )

                async def accepted():
                    sent.append((args, kwargs))
                    return SendResult(True)

                return await delivery_ledger.run(units[0], accepted)

        restarted = DeliveryStore(self.path)
        rows = await claim_deliveries(restarted, {"telegram"})

        self.assertEqual(sent, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["obligation_id"], obligation_id)
        self.assertFalse(rows[0]["needs_marker"])

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
                    ("42", content),
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
            async def send(self, *args, delivery_ledger, **kwargs):
                units = await delivery_ledger.prepare(
                    [("text", delivery_fingerprint("telegram-text", args[1]))]
                )

                async def accepted():
                    sent.append((args, kwargs))
                    return SendResult(True)

                return await delivery_ledger.run(units[0], accepted)

        recovered = await recover_deliveries(
            DeliveryStore(self.path),
            {"telegram": Channel()},
        )

        self.assertEqual(recovered, 1)
        self.assertEqual(
            sent,
            [
                (
                    ("42", content, "777"),
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
        states = []

        class Channel:
            async def send(self, chat_id, text, *, delivery_ledger):
                states.append(self_owner.state(obligation_id))
                units = await delivery_ledger.prepare(
                    [("text", delivery_fingerprint("whatsapp-text", text))]
                )
                states.append(self_owner.state(obligation_id))

                async def accepted():
                    states.append(self_owner.state(obligation_id))
                    sent.append((chat_id, text))
                    return SendResult(True)

                return await delivery_ledger.run(units[0], accepted)

        restarted = DeliveryStore(self.path)
        self_owner = self
        rows = await claim_deliveries(restarted, {"whatsapp"})
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["unitized"])
        self.assertEqual(self.state(obligation_id), "pending")
        self.assertEqual(
            await redeliver_claimed_deliveries(
                restarted,
                {"whatsapp": Channel()},
                rows,
            ),
            1,
        )
        self.assertEqual(sent, [("chat", content)])
        self.assertEqual(states, ["pending", "pending", "attempting"])

    async def test_zero_unit_pending_claim_is_reclaimable_after_claimer_crash(self):
        content = "still known unsent"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content=content,
        )

        crashed_owner = DeliveryStore(self.path)
        first = await claim_deliveries(crashed_owner, {"telegram"})
        self.assertEqual(len(first), 1)
        self.assertFalse(first[0]["unitized"])
        self.assertEqual(self.state(obligation_id), "pending")

        next_runtime = DeliveryStore(self.path)
        second = await claim_deliveries(next_runtime, {"telegram"})
        self.assertEqual(len(second), 1)
        self.assertFalse(second[0]["unitized"])
        self.assertEqual(self.state(obligation_id), "pending")

    async def test_repeated_zero_unit_claim_crashes_do_not_consume_attempts(self):
        content = "never reached planning"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content=content,
        )

        for _ in range(MAX_ATTEMPTS + 2):
            runtime = DeliveryStore(self.path)
            rows = await claim_deliveries(runtime, {"telegram"})
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["unitized"])
            self.assertEqual(rows[0]["attempts"], 0)
            self.assertEqual(self.state(obligation_id), "pending")

    async def test_repeated_unitized_pending_claim_crashes_do_not_consume_attempts(self):
        content = "planned but never activated"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content=content,
        )
        ledger = DeliveryUnitLedger(self.store, obligation_id)
        await ledger.prepare(
            [("text", delivery_fingerprint("telegram-text", content))]
        )

        for _ in range(MAX_ATTEMPTS + 2):
            runtime = DeliveryStore(self.path)
            rows = await claim_deliveries(runtime, {"telegram"})
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["unitized"])
            self.assertEqual(rows[0]["attempts"], 0)
            self.assertEqual(self.state(obligation_id), "pending")

    async def test_pre_network_attempting_restart_claims_do_not_count(self):
        content = "activated but never reached the network"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content=content,
        )
        ledger = DeliveryUnitLedger(self.store, obligation_id)
        units = await ledger.prepare(
            [("text", delivery_fingerprint("telegram-text", content))]
        )
        self.assertTrue(self.store.activate_unit_plan(obligation_id))
        baseline = float(self.row(obligation_id)["updated_at"])

        runtime = self.store
        for index in range(MAX_ATTEMPTS + 2):
            runtime = DeliveryStore(self.path)
            rows = await claim_deliveries(
                runtime,
                {"telegram"},
                now=baseline + index + 1,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["attempts"], 0)
            self.assertEqual(self.state(obligation_id), "attempting")

        self.assertTrue(
            runtime.mark_unit_attempting(obligation_id, units[0].unit_id)
        )
        self.assertTrue(
            runtime.mark_unit_failed(
                obligation_id,
                units[0].unit_id,
                "safely rejected",
                retry_safe=True,
            )
        )
        counted = await claim_deliveries(
            DeliveryStore(self.path),
            {"telegram"},
            now=baseline + MAX_ATTEMPTS + 10,
        )
        self.assertEqual(len(counted), 1)
        self.assertEqual(counted[0]["attempts"], 1)

    async def test_pre_network_attempting_live_claims_do_not_count(self):
        content = "live activation never reached the network"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content=content,
        )
        ledger = DeliveryUnitLedger(self.store, obligation_id)
        units = await ledger.prepare(
            [("text", delivery_fingerprint("telegram-text", content))]
        )
        self.assertTrue(self.store.activate_unit_plan(obligation_id))
        baseline = float(self.row(obligation_id)["updated_at"])

        for index in range(MAX_ATTEMPTS + 2):
            rows = await claim_live_deliveries(
                self.store,
                {"telegram"},
                now=baseline + index + 1,
                min_age_seconds=0,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["attempts"], 0)
            self.assertEqual(self.state(obligation_id), "attempting")

        self.assertTrue(
            self.store.mark_unit_attempting(obligation_id, units[0].unit_id)
        )
        self.assertTrue(
            self.store.mark_unit_failed(
                obligation_id,
                units[0].unit_id,
                "safely rejected",
                retry_safe=True,
            )
        )
        counted = await claim_live_deliveries(
            self.store,
            {"telegram"},
            now=baseline + MAX_ATTEMPTS + 10,
            min_age_seconds=0,
        )
        self.assertEqual(len(counted), 1)
        self.assertEqual(counted[0]["attempts"], 1)

    async def test_pending_recovery_claim_is_released_after_missing_channel(self):
        content = "release the local planning lease"
        obligation_id = compute_obligation_id("session", "message", content)
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content=content,
        )
        pending_at = float(self.row(obligation_id)["updated_at"])
        first = await claim_live_deliveries(
            self.store,
            {"telegram"},
            now=pending_at + 61,
        )
        self.assertEqual(len(first), 1)

        await redeliver_claimed_deliveries(self.store, {}, first)
        second = await claim_live_deliveries(
            self.store,
            {"telegram"},
            now=pending_at + 122,
        )
        self.assertEqual(len(second), 1)

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
            async def send(self, chat_id, text, *, delivery_ledger):
                units = await delivery_ledger.prepare(
                    [("text", delivery_fingerprint("whatsapp-text", text))]
                )

                async def accepted():
                    sent.append((chat_id, text))
                    return SendResult(True)

                return await delivery_ledger.run(units[0], accepted)

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
            [("chat", content)],
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
        self.assertEqual(self.state(obligation_id), "pending")

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

    async def test_stale_unknown_acceptance_becomes_retained_abandoned(self):
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
        created_at = float(self.row(obligation_id)["created_at"])

        self.assertEqual(
            await claim_deliveries(
                DeliveryStore(self.path),
                {"telegram"},
                now=created_at + STALE_AFTER_SECONDS + 1,
            ),
            [],
        )
        abandoned_at = float(self.row(obligation_id)["updated_at"])
        self.assertEqual(self.state(obligation_id), "abandoned")

        await claim_deliveries(
            DeliveryStore(self.path),
            set(),
            now=abandoned_at + RETENTION_SECONDS - 1,
        )
        self.assertEqual(self.state(obligation_id), "abandoned")
        await claim_deliveries(
            DeliveryStore(self.path),
            set(),
            now=abandoned_at + RETENTION_SECONDS + 1,
        )
        self.assertIsNone(self.row(obligation_id))

    async def test_stale_inflight_unit_is_abandoned_without_changing_its_proof(self):
        obligation_id = compute_obligation_id("session", "message", "answer")
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="whatsapp",
            chat_id="chat",
            thread_id="",
            content="answer",
        )
        units = [
            DeliveryUnit(
                unit_id="unit-1",
                position=0,
                kind="text",
                fingerprint="exact-fingerprint",
            )
        ]
        self.store.record_units(obligation_id, units)
        self.store.mark_attempting(obligation_id)
        self.store.mark_unit_attempting(obligation_id, units[0].unit_id)
        created_at = float(self.row(obligation_id)["created_at"])

        self.assertEqual(
            await claim_deliveries(
                DeliveryStore(self.path),
                {"whatsapp"},
                now=created_at + STALE_AFTER_SECONDS + 1,
            ),
            [],
        )
        self.assertEqual(self.state(obligation_id), "abandoned")
        unit = self.unit_rows(obligation_id)[0]
        self.assertEqual(unit["state"], "attempting")
        self.assertEqual(unit["fingerprint"], "exact-fingerprint")

    async def test_stale_retry_after_cannot_bypass_the_recovery_window(self):
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
        self.store.mark_failed(
            obligation_id,
            "rate limited",
            retry_safe=True,
            retry_after=STALE_AFTER_SECONDS * 2,
        )
        created_at = float(self.row(obligation_id)["created_at"])

        self.assertEqual(
            await claim_deliveries(
                DeliveryStore(self.path),
                {"telegram"},
                now=created_at + STALE_AFTER_SECONDS + 1,
            ),
            [],
        )
        self.assertEqual(self.state(obligation_id), "abandoned")

    async def test_capacity_pressure_keeps_a_recent_abandoned_dedupe_fence(self):
        obligation_id = "uncertain"
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
        created_at = float(self.row(obligation_id)["created_at"])
        await claim_live_deliveries(
            self.store,
            set(),
            now=created_at + STALE_AFTER_SECONDS + 1,
            min_age_seconds=0,
        )
        self.assertEqual(self.state(obligation_id), "abandoned")

        stamp = time.time()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executemany(
                "INSERT INTO delivery_obligations"
                " (obligation_id, session_key, platform, chat_id, thread_id,"
                " reply_to, content, state, attempts, created_at, updated_at,"
                " owner_token, last_error, retry_safe, next_attempt_at)"
                " VALUES (?, ?, 'telegram', '42', NULL, NULL, 'done',"
                " 'delivered', 1, ?, ?, 'old-owner', NULL, 0, 0)",
                [
                    (f"delivered-{index}", f"session-{index}", stamp, stamp)
                    for index in range(MAX_ROWS - 1)
                ],
            )
            connection.commit()

        self.store.record(
            obligation_id="new-pending",
            session_key="new-session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content="new answer",
        )

        self.assertEqual(self.state(obligation_id), "abandoned")
        self.assertEqual(self.state("new-pending"), "pending")

    async def test_stale_pending_work_remains_recoverable(self):
        obligation_id = "known-unsent"
        self.store.record(
            obligation_id=obligation_id,
            session_key="session",
            platform="telegram",
            chat_id="42",
            thread_id="",
            content="answer",
        )
        created_at = float(self.row(obligation_id)["created_at"])

        await claim_live_deliveries(
            self.store,
            set(),
            now=created_at + STALE_AFTER_SECONDS * 10,
            min_age_seconds=0,
        )
        self.assertEqual(self.state(obligation_id), "pending")

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
        self.assertEqual(self.state(obligation_id), "pending")

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
