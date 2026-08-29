"""Conversations that outlive the process.

A restart is invisible from a phone. The person writes again, and the agent
either remembers or does not — so the thing worth testing is not that rows are
written, it is that a second Agent over the same file picks the conversation up
where the first one left it.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import ANY, AsyncMock, patch

from pilotage import media
from pilotage.agent import Agent, TurnResult
from pilotage.codex import stream as codex_stream
from pilotage.config import Config
from pilotage.cron.jobs import timezone_for_name
from pilotage.delivery import DeliveryStore
from pilotage.history import (
    ConversationError,
    ConversationStore,
    session_workspace_path,
)
from pilotage.i18n import t
from pilotage.main import _recover_interrupted_turns, _retire_unknown_turn_claims


class StoreTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "state" / "conversations.db"
        self.store = ConversationStore(self.path)

    def test_turns_come_back_in_the_order_they_were_said(self):
        self.store.append("chat", [("user", "first"), ("assistant", "second")])
        self.store.append("chat", [("user", "third"), ("assistant", "fourth")])
        self.assertEqual(
            self.store.load("chat", 10),
            [("user", "first"), ("assistant", "second"), ("user", "third"), ("assistant", "fourth")],
        )

    def test_only_the_last_turns_are_read_back(self):
        self.store.append("chat", [("user", str(n)) for n in range(10)])
        self.assertEqual(self.store.load("chat", 3), [("user", "7"), ("user", "8"), ("user", "9")])

    def test_chats_do_not_see_each_other(self):
        self.store.append("a", [("user", "mine")])
        self.store.append("b", [("user", "theirs")])
        self.assertEqual(self.store.load("a", 10), [("user", "mine")])

    def test_an_unknown_chat_is_empty_not_an_error(self):
        self.assertEqual(self.store.load("nobody", 10), [])

    def test_the_file_is_created_on_first_use(self):
        self.assertFalse(self.path.exists())
        self.store.append("chat", [("user", "hello")])
        self.assertTrue(self.path.exists())

    def test_a_new_session_hides_what_came_before(self):
        self.store.append("chat", [("user", "old business")])
        self.store.new_session("chat")
        self.assertEqual(self.store.load("chat", 10), [])
        self.store.append("chat", [("user", "new business")])
        self.assertEqual(self.store.load("chat", 10), [("user", "new business")])

    def test_a_new_session_deletes_nothing(self):
        """Ending a conversation has to stay cheap; the rows are still there."""
        self.store.append("chat", [("user", "old business")])
        self.store.new_session("chat")
        with closing(sqlite3.connect(self.path)) as connection:
            kept = connection.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        self.assertEqual(kept, 1)

    def test_sessions_keep_stacking(self):
        for round_number in range(3):
            self.store.append("chat", [("user", f"round {round_number}")])
            self.store.new_session("chat")
        self.store.append("chat", [("user", "latest")])
        self.assertEqual(self.store.load("chat", 10), [("user", "latest")])

    def test_current_session_tracks_each_durable_boundary(self):
        self.assertEqual(self.store.current_session("chat"), 1)
        self.store.new_session("chat")
        self.assertEqual(self.store.current_session("chat"), 2)

    def test_idle_policy_starts_a_fresh_durable_session(self):
        self.store.append("chat", [("user", "old context")])
        reset = self.store.prepare_session(
            "chat",
            mode="idle",
            idle_minutes=30,
            now=time.time() + 3600,
        )
        self.assertEqual((reset.reason, reset.had_activity), ("idle", True))
        self.assertEqual(self.store.load("chat", 10), [])

    def test_daily_policy_uses_the_configured_local_boundary(self):
        before = datetime(2026, 1, 2, 3, 0, tzinfo=timezone.utc).timestamp()
        after = datetime(2026, 1, 2, 5, 0, tzinfo=timezone.utc).timestamp()
        self.store.append("chat", [("user", "before boundary")])
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "UPDATE chats SET last_active = ? WHERE chat_id = 'chat'",
                (before,),
            )
            connection.commit()
        reset = self.store.prepare_session(
            "chat",
            mode="daily",
            at_hour=4,
            now=after,
            tzinfo=timezone.utc,
        )
        self.assertEqual(reset.reason, "daily")

    def test_prune_removes_only_old_ended_sessions(self):
        self.store.append("chat", [("user", "old")])
        self.store.new_session("chat")
        self.store.append("chat", [("user", "current")])
        cutoff_now = time.time() + (100 * 86400)
        removed = self.store.prune_old_sessions(90, now=cutoff_now)
        self.assertEqual(removed, 1)
        self.assertEqual(self.store.load("chat", 10), [("user", "current")])
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                "SELECT session, content FROM turns ORDER BY session"
            ).fetchall()
        self.assertEqual(rows, [(2, "current")])

    def test_prune_removes_only_the_expired_session_workspace(self):
        self.store.append("chat", [("user", "old")])
        self.store.new_session("chat")
        self.store.append("chat", [("user", "current")])
        workspace = self.path.parent / "workspace"
        old = session_workspace_path(workspace, "chat", 1)
        current = session_workspace_path(workspace, "chat", 2)
        unrelated = workspace / "sessions" / "unrelated" / "session-1"
        for root in (old, current, unrelated):
            (root / "inputs").mkdir(parents=True)
            (root / "inputs" / "attachment.txt").write_text("keep or prune")

        removed = self.store.prune_old_sessions(
            90,
            now=time.time() + (100 * 86400),
            workspace_roots=(workspace,),
        )

        self.assertEqual(removed, 1)
        self.assertFalse(old.exists())
        self.assertTrue(current.is_dir())
        self.assertTrue(unrelated.is_dir())

    def test_workspace_failure_does_not_undo_the_database_prune(self):
        self.store.append("chat", [("user", "old")])
        self.store.new_session("chat")
        self.store.append("chat", [("user", "current")])
        workspace = self.path.parent / "workspace"
        old = session_workspace_path(workspace, "chat", 1)
        old.mkdir(parents=True)

        with patch("pilotage.history.shutil.rmtree", side_effect=OSError):
            removed = self.store.prune_old_sessions(
                90,
                now=time.time() + (100 * 86400),
                workspace_roots=(workspace,),
            )

        self.assertEqual(removed, 1)
        self.assertTrue(old.is_dir())
        self.assertEqual(self.store.load("chat", 10), [("user", "current")])

    def test_only_opaque_compaction_checkpoints_are_replayable(self):
        self.store.append_with_replay(
            "chat",
            [
                (
                    "assistant",
                    "answer",
                    [
                        {"type": "reasoning", "encrypted_content": "thought"},
                        {"type": "compaction", "encrypted_content": "checkpoint"},
                        {"type": "compaction", "encrypted_content": ""},
                    ],
                )
            ],
        )
        self.assertEqual(
            self.store.load_with_replay("chat"),
            [
                (
                    "assistant",
                    "answer",
                    [{"type": "compaction", "encrypted_content": "checkpoint"}],
                )
            ],
        )

    def test_an_existing_database_is_migrated_without_losing_turns(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "CREATE TABLE turns ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL, "
                "session INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, "
                "written_at REAL NOT NULL)"
            )
            connection.execute(
                "INSERT INTO turns (chat_id, session, role, content, written_at)"
                " VALUES ('chat', 1, 'user', 'before migration', 1.0)"
            )
            connection.commit()

        self.store.append_with_replay(
            "chat",
            [("assistant", "after migration", [{"type": "compaction", "encrypted_content": "cp"}])],
        )
        self.assertEqual(
            self.store.load("chat", 10),
            [("user", "before migration"), ("assistant", "after migration")],
        )


    def test_a_broken_file_fails_the_history_write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"this is not a database")
        with (
            self.assertLogs("pilotage.history", level="WARNING"),
            self.assertRaises(ConversationError),
        ):
            self.store.append("chat", [("user", "hello")])
        with (
            self.assertLogs("pilotage.history", level="WARNING"),
            self.assertRaises(ConversationError),
        ):
            self.store.load("chat", 10)

    def test_active_turn_records_its_delivery_origin_and_iteration(self):
        self.store.begin_turn(
            "chat",
            "do it",
            origin={
                "channel": "telegram",
                "chat_id": "42",
                "thread_id": "7",
                "reply_to": "99",
            },
        )
        items = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "todo",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            },
        ]
        self.store.checkpoint_turn(
            "chat",
            "do it",
            items[:1],
            phase="tool_requested",
            iteration=3,
        )
        self.store.checkpoint_turn(
            "chat",
            "do it",
            items,
            phase="tool_completed",
            iteration=3,
        )

        active = self.store.list_active_turns()

        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].trajectory, items)
        self.assertEqual(active[0].iteration, 3)
        self.assertEqual(active[0].origin["reply_to"], "99")

    def test_final_answer_stays_fenced_until_explicit_completion(self):
        claim_id = "a" * 64
        replay = [{"type": "compaction", "encrypted_content": "opaque"}]
        self.store.begin_turn(
            "chat",
            "accepted",
            origin={"channel": "whatsapp", "chat_id": "123@c.us"},
            claim_ids=[claim_id],
        )

        self.store.checkpoint_answer(
            "chat",
            "accepted",
            "Exact answer.",
            replay,
            terminal_completed=True,
        )

        active = self.store.list_active_turns()[0]
        self.assertEqual(active.phase, "answer_ready")
        self.assertEqual(active.answer_content, "Exact answer.")
        self.assertEqual(active.answer_replay, replay)
        self.assertEqual(active.claim_ids, [claim_id])
        self.assertTrue(active.terminal_completed)
        self.assertEqual(self.store.load("chat", 10), [])

        completed = self.store.complete_ready_turn("chat")

        self.assertEqual(
            completed,
            ("accepted", "Exact answer.", replay, True),
        )
        self.assertEqual(self.store.list_active_turns(), [])
        self.assertEqual(
            self.store.load("chat", 10),
            [("user", "accepted"), ("assistant", "Exact answer.")],
        )

    def test_answer_ready_cannot_be_corrupted_into_an_unknown_phase(self):
        self.store.begin_turn("chat", "accepted")
        self.store.checkpoint_answer(
            "chat",
            "accepted",
            "Exact answer.",
        )
        active = self.store.list_active_turns()[0]

        with self.assertRaisesRegex(ConversationError, "cannot be downgraded"):
            self.store.mark_turn_unknown(active)

        for _ in range(2):
            active = ConversationStore(self.path).list_active_turns()[0]
            self.assertEqual(active.phase, "answer_ready")
            self.assertEqual(active.answer_content, "Exact answer.")

    def test_interrupted_requested_tool_is_fenced_as_unknown(self):
        self.store.begin_turn("chat", "possibly acted")
        self.store.checkpoint_turn(
            "chat",
            "possibly acted",
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "todo",
                    "arguments": "{}",
                }
            ],
            phase="tool_requested",
            iteration=1,
        )
        active = self.store.list_active_turns()[0]

        self.store.mark_turn_unknown(active)

        self.assertEqual(self.store.list_active_turns()[0].phase, "unknown")
        with self.assertRaisesRegex(ConversationError, "previous turn"):
            self.store.begin_turn("chat", "new request")

    def test_corrupt_active_trajectory_fails_closed(self):
        self.store.begin_turn("chat", "do it")
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("UPDATE active_turns SET trajectory = 'not-json'")
            connection.commit()

        with (
            self.assertLogs("pilotage.history", level="WARNING"),
            self.assertRaises(ConversationError),
        ):
            self.store.list_active_turns()

    def test_active_tool_trajectory_is_removed_with_the_completed_turn(self):
        self.store.begin_turn("chat", "do it")
        items = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "todo",
                "arguments": "{}",
            }
        ]
        self.store.checkpoint_turn(
            "chat", "do it", items, phase="tool_requested"
        )
        with closing(sqlite3.connect(self.path)) as connection:
            phase, trajectory = connection.execute(
                "SELECT phase, trajectory FROM active_turns"
            ).fetchone()
        self.assertEqual(phase, "tool_requested")
        self.assertEqual(json.loads(trajectory), items)

        with self.assertRaisesRegex(ConversationError, "no longer safe"):
            self.store.complete_turn("chat", "do it", "Done too early.")
        self.assertEqual(self.store.load("chat", 10), [])

        completed_items = [
            *items,
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            },
        ]
        self.store.checkpoint_turn(
            "chat",
            "do it",
            completed_items,
            phase="tool_completed",
        )
        self.store.complete_turn("chat", "do it", "Done.")

        with closing(sqlite3.connect(self.path)) as connection:
            active = connection.execute(
                "SELECT COUNT(*) FROM active_turns"
            ).fetchone()[0]
        self.assertEqual(active, 0)
        self.assertEqual(
            self.store.load("chat", 10),
            [("user", "do it"), ("assistant", "Done.")],
        )

    def test_new_session_explicitly_abandons_an_incomplete_turn(self):
        self.store.begin_turn("chat", "possibly acted")
        self.store.checkpoint_turn(
            "chat",
            "possibly acted",
            [{"type": "function_call", "call_id": "call_1"}],
            phase="tool_requested",
        )

        self.store.new_session("chat")
        self.store.begin_turn("chat", "fresh")
        self.store.complete_turn("chat", "fresh", "Ready.")

        self.assertEqual(
            self.store.load("chat", 10),
            [("user", "fresh"), ("assistant", "Ready.")],
        )

    def test_automatic_reset_never_discards_an_incomplete_tool_turn(self):
        self.store.begin_turn("chat", "possibly acted")
        self.store.checkpoint_turn(
            "chat",
            "possibly acted",
            [{"type": "function_call", "call_id": "call_1"}],
            phase="tool_requested",
        )

        reset = self.store.prepare_session(
            "chat",
            mode="idle",
            idle_minutes=1,
            now=time.time() + 3600,
        )

        self.assertIsNone(reset)
        self.assertEqual(self.store.current_session("chat"), 1)
        with self.assertRaisesRegex(ConversationError, "previous turn"):
            self.store.begin_turn("chat", "new request")


class RestartTests(unittest.IsolatedAsyncioTestCase):
    """The same file, a second Agent — which is what an upgrade looks like."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "conversations.db"

    def _agent(self) -> Agent:
        agent = Agent(Config.load(), ConversationStore(self.path))

        async def _stream_once(request, *, force_refresh, ttfb_timeout, idle_timeout):
            self.sent = request
            return codex_stream.StreamResult(text="Answered.")

        agent._stream_once = _stream_once
        return agent

    async def test_unknown_tool_turn_retires_input_but_keeps_the_unknown_fence(self):
        store = ConversationStore(self.path)
        claim_id = "f" * 64
        store.begin_turn(
            "chat",
            "possibly acted",
            origin={"channel": "telegram", "chat_id": "42", "reply_to": "9"},
            claim_ids=[claim_id],
        )
        store.checkpoint_turn(
            "chat",
            "possibly acted",
            [{"type": "function_call", "call_id": "call_1"}],
            phase="tool_requested",
        )
        active = store.list_active_turns()[0]
        store.mark_turn_unknown(active)
        active = store.list_active_turns()[0]
        completed_claims = []

        class FakeChannel:
            failure = None

            def persist_completed_claims(self, claim_ids):
                completed_claims.extend(claim_ids)

            def _fail(self, message):
                self.failure = message

        channel = FakeChannel()
        with patch(
            "pilotage.main._deliver_recovered_turn",
            new=AsyncMock(return_value=True),
        ) as deliver:
            retired = await _retire_unknown_turn_claims(
                [active],
                channels={"telegram": channel},
                delivery_store=DeliveryStore(self.path.parent / "delivery.db"),
                notices={"telegram": "Verify it, then use /new."},
            )

        self.assertEqual(retired, 1)
        self.assertEqual(completed_claims, [claim_id])
        deliver.assert_awaited_once_with(
            ANY,
            channel,
            active,
            "Verify it, then use /new.",
        )
        self.assertEqual(store.list_active_turns()[0].phase, "unknown")
        with self.assertRaisesRegex(ConversationError, "previous turn"):
            store.begin_turn("chat", "must remain fenced")

    async def test_a_conversation_survives_the_process(self):
        first = self._agent()
        await first.respond("chat", "my name is Sam")

        second = self._agent()
        await second.respond("chat", "what is my name?")

        said = [item.get("content") for item in self.sent["input"]]
        self.assertIn("my name is Sam", said)
        self.assertIn("Answered.", said)

    async def test_a_failed_history_read_is_retried_without_losing_continuity(self):
        store = ConversationStore(self.path)
        store.append(
            "chat",
            [("user", "remember the blue contract"), ("assistant", "Remembered.")],
        )
        agent = self._agent()
        original_load = agent._store.load_with_replay
        attempts = 0

        def load_once_then_succeed(chat_id, limit=None):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConversationError("temporary read failure")
            return original_load(chat_id, limit)

        with patch.object(
            agent._store,
            "load_with_replay",
            side_effect=load_once_then_succeed,
        ):
            with self.assertRaisesRegex(ConversationError, "temporary read failure"):
                await agent.respond("chat", "first attempt")
            self.assertNotIn("chat", agent._restored)

            await agent.respond("chat", "retry")

        self.assertEqual(attempts, 2)
        said = [item.get("content") for item in self.sent["input"]]
        self.assertIn("remember the blue contract", said)
        self.assertIn("Remembered.", said)

    async def test_a_started_turn_is_resumed_after_restart(self):
        store = ConversationStore(self.path)
        store.begin_turn(
            "chat",
            "accepted before crash",
            origin={"channel": "telegram", "chat_id": "42", "reply_to": "9"},
        )
        second = self._agent()

        result = await second.recover_turn(store.list_active_turns()[0])

        self.assertEqual(result.text, "Answered.")
        self.assertEqual(store.list_active_turns(), [])
        self.assertEqual(
            store.load("chat", 2),
            [("user", "accepted before crash"), ("assistant", "Answered.")],
        )

    async def test_a_staged_answer_survives_restart_without_calling_the_model(self):
        first = self._agent()
        result = await first.respond_result(
            "chat",
            "accepted before delivery",
            defer_completion=True,
        )
        self.assertEqual(result.text, "Answered.")

        store = ConversationStore(self.path)
        active = store.list_active_turns()[0]
        self.assertEqual(active.phase, "answer_ready")
        self.assertEqual(active.answer_content, "Answered.")

        second = self._agent()
        second._stream_once = AsyncMock(
            side_effect=AssertionError("a staged answer must not be regenerated")
        )
        recovered = await second.recover_turn(active, defer_completion=True)

        self.assertEqual(recovered.text, "Answered.")
        second._stream_once.assert_not_awaited()
        self.assertEqual(store.list_active_turns()[0].phase, "answer_ready")

        await second.finalize_ready_turn("chat")

        self.assertEqual(store.list_active_turns(), [])
        self.assertEqual(
            store.load("chat", 2),
            [
                ("user", "accepted before delivery"),
                ("assistant", "Answered."),
            ],
        )

    async def test_a_staged_answer_survives_even_if_its_old_image_is_gone(self):
        workspace = self.path.parent / "workspace"
        source = self.path.parent / "source.png"
        source.write_bytes(b"image-pixels")
        attachment = media.Attachment(
            path=source,
            mime="image/png",
            media_type="image",
        )
        first = self._agent()
        first._context_cwd = workspace
        await first.respond_result(
            "chat",
            "inspect",
            [attachment],
            defer_completion=True,
        )

        store = ConversationStore(self.path)
        active = store.list_active_turns()[0]
        inputs = first._session_working_directory("chat") / "inputs"
        (inputs / active.image_manifest[0]["path"]).unlink()

        second = self._agent()
        second._context_cwd = workspace
        second._stream_once = AsyncMock(
            side_effect=AssertionError("a staged answer must not be regenerated")
        )
        with self.assertLogs("pilotage.agent", level="WARNING"):
            recovered = await second.recover_turn(active, defer_completion=True)

        self.assertEqual(recovered.text, "Answered.")
        second._stream_once.assert_not_awaited()
        await second.finalize_ready_turn("chat")

    async def test_an_empty_completed_response_is_normalized_before_staging(self):
        agent = self._agent()

        async def _empty_response(
            request, *, force_refresh, ttfb_timeout, idle_timeout
        ):
            return codex_stream.StreamResult(
                text="",
                terminal_completed=True,
            )

        agent._stream_once = _empty_response
        result = await agent.respond_result(
            "chat",
            "answer me",
            defer_completion=True,
        )

        active = ConversationStore(self.path).list_active_turns()[0]
        self.assertTrue(result.text)
        self.assertEqual(active.answer_content, result.text)

    async def test_a_model_failure_is_staged_as_the_exact_failure_reply(self):
        agent = self._agent()
        agent._stream_once = AsyncMock(
            side_effect=codex_stream.CodexStreamError("backend unavailable")
        )

        with self.assertLogs("pilotage.agent", level="WARNING"):
            result = await agent.respond_result(
                "chat",
                "answer me",
                defer_completion=True,
            )

        failure = t("runtime.failure", agent._config.language)
        active = ConversationStore(self.path).list_active_turns()[0]
        self.assertEqual(result.text, failure)
        self.assertFalse(result.terminal_completed)
        self.assertEqual(active.phase, "answer_ready")
        self.assertEqual(active.answer_content, failure)
        self.assertFalse(active.terminal_completed)

        await agent.finalize_ready_turn("chat")
        self.assertEqual(
            ConversationStore(self.path).load("chat", 2),
            [("user", "answer me"), ("assistant", failure)],
        )

    async def test_new_refuses_while_the_answer_delivery_fence_is_open(self):
        agent = self._agent()
        await agent.respond_result(
            "chat",
            "finish this first",
            defer_completion=True,
        )

        self.assertFalse(await agent.forget("chat"))
        self.assertEqual(ConversationStore(self.path).current_session("chat"), 1)

        await agent.finalize_ready_turn("chat")
        self.assertTrue(await agent.forget("chat"))

        store = ConversationStore(self.path)
        self.assertEqual(store.current_session("chat"), 2)
        self.assertEqual(store.list_active_turns(), [])
        self.assertEqual(store.load("chat", 10), [])

    async def test_failed_recovery_never_leaves_new_waiting(self):
        store = ConversationStore(self.path)
        store.begin_turn("chat", "recover me")
        agent = self._agent()
        agent._run_turn = AsyncMock(side_effect=RuntimeError("unexpected failure"))

        with self.assertRaisesRegex(RuntimeError, "unexpected failure"):
            await agent.recover_turn(
                store.list_active_turns()[0],
                defer_completion=True,
            )

        await asyncio.wait_for(agent.forget("chat"), timeout=1)
        self.assertEqual(store.current_session("chat"), 2)
        self.assertEqual(store.list_active_turns(), [])

    async def test_a_completed_tool_checkpoint_resumes_without_rerunning_it(self):
        store = ConversationStore(self.path)
        store.begin_turn(
            "chat",
            "continue safely",
            origin={"channel": "whatsapp", "chat_id": "212600000000"},
        )
        items = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "todo",
                "arguments": '{"todos": []}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": '{"todos": []}',
            },
        ]
        store.checkpoint_turn(
            "chat",
            "continue safely",
            items[:1],
            phase="tool_requested",
            iteration=1,
        )
        store.checkpoint_turn(
            "chat",
            "continue safely",
            items,
            phase="tool_completed",
            iteration=1,
        )
        second = self._agent()
        second._registry.dispatch = AsyncMock(
            side_effect=AssertionError("completed tool must not run again")
        )

        await second.recover_turn(store.list_active_turns()[0])

        replay = self.sent["input"]
        self.assertIn(items[0], replay)
        self.assertIn(items[1], replay)
        second._registry.dispatch.assert_not_awaited()

    async def test_failed_model_recovery_delivers_a_staged_tool_warning(self):
        store = ConversationStore(self.path)
        claim_id = "b" * 64
        store.begin_turn(
            "chat",
            "continue safely",
            origin={"channel": "telegram", "chat_id": "42", "reply_to": "9"},
            claim_ids=[claim_id],
        )
        items = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "todo",
                "arguments": '{"todos": []}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": '{"todos": []}',
            },
        ]
        store.checkpoint_turn(
            "chat",
            "continue safely",
            items[:1],
            phase="tool_requested",
            iteration=1,
        )
        store.checkpoint_turn(
            "chat",
            "continue safely",
            items,
            phase="tool_completed",
            iteration=1,
        )
        agent = self._agent()
        agent._stream_once = AsyncMock(
            side_effect=codex_stream.CodexStreamError("backend unavailable")
        )
        agent._registry.dispatch = AsyncMock(
            side_effect=AssertionError("completed tool must not run again")
        )
        completed_claims = []

        class FakeChannel:
            failure = None

            def persist_completed_claims(self, claim_ids):
                completed_claims.extend(claim_ids)

            def _fail(self, message):
                self.failure = message

        channel = FakeChannel()
        warning = t("runtime.interrupted_unknown", agent._config.language)

        async def deliver_staged(_delivery_store, _channel, _active, text):
            staged = store.list_active_turns()[0]
            self.assertEqual(staged.phase, "answer_ready")
            self.assertEqual(staged.answer_content, warning)
            self.assertEqual(text, warning)
            return True

        with (
            self.assertLogs("pilotage.agent", level="WARNING"),
            patch(
                "pilotage.main._deliver_recovered_turn",
                new=AsyncMock(side_effect=deliver_staged),
            ) as deliver,
        ):
            recovered = await _recover_interrupted_turns(
                store.list_active_turns(),
                agents={"telegram": agent},
                channels={"telegram": channel},
                conversation_store=store,
                delivery_store=DeliveryStore(self.path.parent / "delivery.db"),
                fenced_turn_sessions=set(),
            )

        self.assertEqual(recovered, 1)
        self.assertEqual(completed_claims, [claim_id])
        self.assertIsNone(channel.failure)
        self.assertEqual(store.list_active_turns(), [])
        self.assertEqual(
            store.load("chat", 2),
            [("user", "continue safely"), ("assistant", warning)],
        )
        deliver.assert_awaited_once()
        agent._registry.dispatch.assert_not_awaited()
        await asyncio.wait_for(agent.forget("chat"), timeout=1)

    async def test_a_requested_tool_checkpoint_refuses_automatic_recovery(self):
        store = ConversationStore(self.path)
        store.begin_turn("chat", "possibly acted")
        store.checkpoint_turn(
            "chat",
            "possibly acted",
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "todo",
                    "arguments": "{}",
                }
            ],
            phase="tool_requested",
            iteration=1,
        )
        second = self._agent()

        with self.assertRaisesRegex(ConversationError, "ambiguous"):
            await second.recover_turn(store.list_active_turns()[0])

    async def test_a_mismatched_completed_checkpoint_fails_closed(self):
        store = ConversationStore(self.path)
        store.begin_turn("chat", "corrupt")
        requested = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "todo",
                "arguments": "{}",
            }
        ]
        store.checkpoint_turn(
            "chat",
            "corrupt",
            requested,
            phase="tool_requested",
            iteration=1,
        )
        store.checkpoint_turn(
            "chat",
            "corrupt",
            [
                *requested,
                {
                    "type": "function_call_output",
                    "call_id": "call_other",
                    "output": "ok",
                },
            ],
            phase="tool_completed",
            iteration=1,
        )

        with self.assertRaisesRegex(ConversationError, "do not match"):
            await self._agent().recover_turn(store.list_active_turns()[0])

    async def test_an_interleaved_completed_checkpoint_fails_closed(self):
        store = ConversationStore(self.path)
        store.begin_turn("chat", "corrupt")
        requested = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "todo",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "todo",
                "arguments": "{}",
            },
        ]
        store.checkpoint_turn(
            "chat",
            "corrupt",
            requested,
            phase="tool_requested",
            iteration=1,
        )
        store.checkpoint_turn(
            "chat",
            "corrupt",
            [
                *requested,
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                },
                {
                    "type": "function_call",
                    "call_id": "call_3",
                    "name": "todo",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "ok",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_3",
                    "output": "ok",
                },
            ],
            phase="tool_completed",
            iteration=1,
        )

        with self.assertRaisesRegex(ConversationError, "out of order"):
            await self._agent().recover_turn(store.list_active_turns()[0])

    async def test_arbitrary_reasoning_checkpoint_fails_closed(self):
        store = ConversationStore(self.path)
        store.begin_turn("chat", "corrupt")
        requested = [
            {"type": "reasoning", "unexpected": "injected"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "todo",
                "arguments": "{}",
            },
        ]
        store.checkpoint_turn(
            "chat",
            "corrupt",
            requested,
            phase="tool_requested",
            iteration=1,
        )
        store.checkpoint_turn(
            "chat",
            "corrupt",
            [
                *requested,
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                },
            ],
            phase="tool_completed",
            iteration=1,
        )

        with self.assertRaisesRegex(ConversationError, "reasoning checkpoint"):
            await self._agent().recover_turn(store.list_active_turns()[0])

    async def test_startup_recovery_finishes_and_routes_the_interrupted_answer(self):
        store = ConversationStore(self.path)
        claim_id = "a" * 64
        store.begin_turn(
            "chat",
            "accepted",
            origin={"channel": "telegram", "chat_id": "42", "reply_to": "9"},
            claim_ids=[claim_id],
        )
        agent = self._agent()
        completed_claims = []

        class FakeChannel:
            def persist_completed_claims(self, claim_ids):
                completed_claims.extend(claim_ids)

        channel = FakeChannel()
        delivery_store = DeliveryStore(self.path.parent / "delivery.db")

        with patch(
            "pilotage.main._deliver_recovered_turn",
            new=AsyncMock(return_value=True),
        ) as deliver:
            recovered = await _recover_interrupted_turns(
                store.list_active_turns(),
                agents={"telegram": agent},
                channels={"telegram": channel},
                conversation_store=store,
                delivery_store=delivery_store,
                fenced_turn_sessions=set(),
            )

        self.assertEqual(recovered, 1)
        self.assertEqual(store.list_active_turns(), [])
        deliver.assert_awaited_once()
        self.assertIs(deliver.await_args.args[1], channel)
        self.assertEqual(deliver.await_args.args[3], "Answered.")
        self.assertEqual(completed_claims, [claim_id])

    async def test_startup_recovery_returns_the_real_approval_send_result(self):
        store = ConversationStore(self.path)
        store.begin_turn(
            "chat",
            "accepted",
            origin={"channel": "telegram", "chat_id": "42", "reply_to": "9"},
        )
        accepted = object()
        seen = {}

        class FakeChannel:
            failure = None

            async def send(self, *_args, **_kwargs):
                return accepted

            def _fail(self, message):
                self.failure = message

        class FakeAgent:
            _config = Config.load()

            async def recover_turn(
                self,
                active,
                on_notice=None,
                *,
                approval_notify=None,
                defer_completion=False,
            ):
                seen["approval_result"] = await approval_notify("Approve")
                seen["defer_completion"] = defer_completion
                return TurnResult(text="Answered.")

            async def finalize_ready_turn(self, chat_id):
                seen["finalized"] = chat_id

        with patch(
            "pilotage.main._deliver_recovered_turn",
            new=AsyncMock(return_value=True),
        ):
            recovered = await _recover_interrupted_turns(
                store.list_active_turns(),
                agents={"telegram": FakeAgent()},
                channels={"telegram": FakeChannel()},
                conversation_store=store,
                delivery_store=DeliveryStore(self.path.parent / "delivery.db"),
                fenced_turn_sessions=set(),
            )

        self.assertEqual(recovered, 1)
        self.assertIs(seen["approval_result"], accepted)
        self.assertTrue(seen["defer_completion"])
        self.assertEqual(seen["finalized"], "chat")

    async def test_failed_recovery_delivery_keeps_the_exact_answer_checkpoint(self):
        store = ConversationStore(self.path)
        store.begin_turn(
            "chat",
            "accepted",
            origin={"channel": "telegram", "chat_id": "42", "reply_to": "9"},
        )
        agent = self._agent()

        class FakeChannel:
            failure = None

            def _fail(self, message):
                self.failure = message

        channel = FakeChannel()
        with patch(
            "pilotage.main._deliver_recovered_turn",
            new=AsyncMock(return_value=False),
        ):
            recovered = await _recover_interrupted_turns(
                store.list_active_turns(),
                agents={"telegram": agent},
                channels={"telegram": channel},
                conversation_store=store,
                delivery_store=DeliveryStore(self.path.parent / "delivery.db"),
                fenced_turn_sessions=set(),
            )

        self.assertEqual(recovered, 0)
        active = store.list_active_turns()[0]
        self.assertEqual(active.phase, "answer_ready")
        self.assertEqual(active.answer_content, "Answered.")
        self.assertIsNotNone(channel.failure)
        self.assertEqual(store.load("chat", 10), [])

    async def test_the_reasoning_of_a_finished_answer_is_not_replayed(self):
        """It belongs to a response the API no longer has; replaying it can fail."""
        first = self._agent()

        async def _with_reasoning(request, *, force_refresh, ttfb_timeout, idle_timeout):
            return codex_stream.StreamResult(
                text="Answered.",
                reasoning_items=[{"type": "reasoning", "id": "rs_1", "encrypted_content": "x"}],
            )

        first._stream_once = _with_reasoning
        await first.respond("chat", "think hard")

        second = self._agent()
        await second.respond("chat", "again")
        self.assertFalse(any(item.get("type") == "reasoning" for item in self.sent["input"]))

    async def test_a_compaction_checkpoint_survives_the_process(self):
        first = self._agent()

        async def _with_checkpoint(request, *, force_refresh, ttfb_timeout, idle_timeout):
            return codex_stream.StreamResult(
                text="Checkpointed.",
                reasoning_items=[
                    {"type": "compaction", "encrypted_content": "opaque-checkpoint"}
                ],
            )

        first._stream_once = _with_checkpoint
        await first.respond("chat", "keep this goal")

        second = self._agent()
        await second.respond("chat", "continue")

        self.assertEqual(
            self.sent["input"][0],
            {"type": "compaction", "encrypted_content": "opaque-checkpoint"},
        )
        said = [item.get("content") for item in self.sent["input"]]
        self.assertIn("keep this goal", said)


    async def test_a_restart_after_new_does_not_hand_the_conversation_back(self):
        first = self._agent()
        await first.respond("chat", "the old business")
        await first.forget("chat")

        second = self._agent()
        await second.respond("chat", "hello?")
        said = [item.get("content") for item in self.sent["input"]]
        self.assertNotIn("the old business", said)

    async def test_idle_auto_reset_clears_live_context_and_notifies_once(self):
        agent = self._agent()
        object.__setattr__(agent._config, "session_reset_mode", "idle")
        object.__setattr__(agent._config, "session_reset_idle_minutes", 1)
        object.__setattr__(agent._config, "session_reset_notify", True)
        notices = []

        async def notice(text, _replace_id=""):
            notices.append(text)

        await agent.respond("chat", "old context", on_notice=notice)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "UPDATE chats SET last_active = ? WHERE chat_id = 'chat'",
                (time.time() - 300,),
            )
            connection.commit()

        await agent.respond("chat", "new topic", on_notice=notice)

        input_text = [
            str(item.get("content") or "")
            for item in self.sent["input"]
            if isinstance(item, dict)
        ]
        self.assertFalse(any("old context" in text for text in input_text))
        self.assertTrue(any("fresh conversation" in text for text in input_text))
        self.assertEqual(len(notices), 1)
        self.assertEqual(
            self.store_for_path().load("chat", 10),
            [("user", "new topic"), ("assistant", "Answered.")],
        )

    async def test_automatic_reset_uses_the_profile_timezone(self):
        agent = self._agent()
        object.__setattr__(agent._config, "session_reset_mode", "daily")
        object.__setattr__(agent._config, "timezone", "Africa/Casablanca")
        seen = {}
        real_prepare = agent._store.prepare_session

        def capture(*args, **kwargs):
            seen["timezone"] = kwargs["tzinfo"]
            return real_prepare(*args, **kwargs)

        agent._store.prepare_session = capture
        await agent.respond("chat", "hello")

        self.assertEqual(
            getattr(seen["timezone"], "key", None),
            getattr(timezone_for_name("Africa/Casablanca"), "key", None),
        )

    async def test_long_turn_sends_the_configured_generic_heartbeat(self):
        agent = self._agent()
        object.__setattr__(
            agent._config,
            "working_notice_interval_seconds",
            0.01,
        )
        object.__setattr__(
            agent._config,
            "working_notice_text",
            "Still safely working.",
        )
        notices = []

        async def slow_stream(
            request,
            *,
            force_refresh,
            ttfb_timeout,
            idle_timeout,
        ):
            await asyncio.sleep(0.04)
            return codex_stream.StreamResult(text="Answered.")

        async def notice(text, _replace_id=""):
            notices.append(text)

        agent._stream_once = slow_stream
        await agent.respond("chat", "take your time", on_notice=notice)

        self.assertEqual(notices, ["Still safely working. (<1 min)"])

    async def test_restricted_mode_routes_each_session_to_new_exports(self):
        agent = self._agent()
        object.__setattr__(
            agent._config,
            "session_isolated_workspaces",
            True,
        )
        agent._context_cwd = self.path.parent / "workspace"
        agent._fixed_working_directory = False

        self.assertEqual(
            agent.session_workspace_root,
            agent._context_cwd.resolve(strict=False),
        )

        await agent.respond("chat", "make a report")
        first = agent._session_workdirs["chat"][1]
        self.assertTrue((first / "inputs").is_dir())
        self.assertTrue((first / "tmp").is_dir())
        self.assertTrue((first / "exports").is_dir())
        self.assertIn(str(first), self.sent["instructions"])
        self.assertIn(str(first / "exports"), self.sent["instructions"])

        await agent.forget("chat")
        await agent.respond("chat", "start over")
        second = agent._session_workdirs["chat"][1]
        self.assertNotEqual(first, second)
        self.assertEqual(first.name, "session-1")
        self.assertEqual(second.name, "session-2")

    async def test_cancelled_workspace_lookup_cannot_revive_the_old_generation(self):
        agent = self._agent()
        object.__setattr__(
            agent._config,
            "session_isolated_workspaces",
            True,
        )
        agent._context_cwd = self.path.parent / "workspace"
        agent._fixed_working_directory = False
        lookup_started = threading.Event()
        release_lookup = threading.Event()
        real_workspace_path = session_workspace_path

        def delayed_workspace_path(base, chat_id, generation):
            result = real_workspace_path(base, chat_id, generation)
            lookup_started.set()
            release_lookup.wait(timeout=2)
            return result

        with patch(
            "pilotage.agent.session_workspace_path",
            side_effect=delayed_workspace_path,
        ):
            stale_lookup = asyncio.create_task(
                asyncio.to_thread(agent._session_working_directory, "chat")
            )
            started = await asyncio.to_thread(lookup_started.wait, 1)
            self.assertTrue(started)
            stale_lookup.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await stale_lookup

            await agent.forget("chat")
            release_lookup.set()
            for _ in range(100):
                cached = agent._session_workdirs.get("chat")
                if cached is not None:
                    break
                await asyncio.sleep(0.01)
            self.assertIsNotNone(cached)
            assert cached is not None
            self.assertEqual(cached[0], 1)

            fresh = await asyncio.to_thread(
                agent._session_working_directory,
                "chat",
            )

        self.assertEqual(fresh.name, "session-2")
        self.assertEqual(agent._session_workdirs["chat"], (2, fresh))

    async def test_inbound_document_path_is_rewritten_to_session_inputs(self):
        agent = self._agent()
        object.__setattr__(
            agent._config,
            "session_isolated_workspaces",
            True,
        )
        agent._context_cwd = self.path.parent / "workspace"
        agent._fixed_working_directory = False
        cached = self.path.parent / "media" / "source.pdf"
        cached.parent.mkdir()
        cached.write_bytes(b"pdf")
        attachment = media.Attachment(
            path=cached.resolve(),
            mime="application/pdf",
            media_type="document",
            file_name="source.pdf",
        )

        await agent.respond(
            "chat",
            f"The document is saved at: {cached.resolve()}",
            [attachment],
        )

        session_root = agent._session_workdirs["chat"][1]
        staged = list((session_root / "inputs").iterdir())
        self.assertEqual(len(staged), 1)
        user_content = str(self.sent["input"][-1]["content"])
        self.assertIn(str(staged[0]), user_content)
        self.assertNotIn(str(cached.resolve()), user_content)

    def store_for_path(self):
        return ConversationStore(self.path)

    async def test_history_is_read_once_per_chat_not_once_per_message(self):
        await self._agent().respond("chat", "remember this")

        second = self._agent()
        reads = []
        real_load = second._store.load_with_replay

        def _counted(chat_id, limit=None):
            reads.append(chat_id)
            return real_load(chat_id, limit)

        second._store.load_with_replay = _counted
        await second.respond("chat", "one")
        await second.respond("chat", "two")
        self.assertEqual(reads, ["chat"])

    async def test_configured_window_remains_the_non_native_fallback(self):
        first = self._agent()
        object.__setattr__(first._config, "codex_native_compaction", False)
        for n in range(first._config.history_turns + 5):
            await first.respond("chat", f"question {n}")

        second = self._agent()
        object.__setattr__(second._config, "codex_native_compaction", False)
        self.assertEqual(len(second._history.get("chat", [])), 0)
        await second.respond("chat", "and now?")
        self.assertEqual(len(second._history["chat"]), first._history_limit())

    async def test_a_store_that_cannot_be_written_stops_before_the_model(self):
        broken = Agent(Config.load(), ConversationStore(self.path))
        self.path.write_bytes(b"this is not a database")
        called = False

        async def _stream_once(request, *, force_refresh, ttfb_timeout, idle_timeout):
            nonlocal called
            called = True
            return codex_stream.StreamResult(text="must not be returned")

        broken._stream_once = _stream_once
        with (
            self.assertLogs("pilotage.history", level="WARNING"),
            self.assertRaises(ConversationError),
        ):
            await broken.respond("chat", "hello")
        self.assertFalse(called)


class EphemeralStoreTests(unittest.TestCase):

    def test_a_store_without_a_path_keeps_nothing(self):
        store = ConversationStore(path=None)
        store.append("cli", [("user", "still there?"), ("assistant", "Yes.")])
        store.new_session("cli")
        self.assertEqual(store.load("cli", 10), [])

if __name__ == "__main__":
    unittest.main()
