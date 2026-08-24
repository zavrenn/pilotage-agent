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
import time
import unittest
from contextlib import closing, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from pilotage import main, media
from pilotage.agent import Agent
from pilotage.codex import stream as codex_stream
from pilotage.config import Config
from pilotage.cron.jobs import timezone_for_name
from pilotage.history import (
    ConversationError,
    ConversationStore,
    session_workspace_path,
)


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
        with self.assertLogs("pilotage.history", level="WARNING"):
            self.assertEqual(self.store.load("chat", 10), [])

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

    async def test_a_conversation_survives_the_process(self):
        first = self._agent()
        await first.respond("chat", "my name is Sam")

        second = self._agent()
        await second.respond("chat", "what is my name?")

        said = [item.get("content") for item in self.sent["input"]]
        self.assertIn("my name is Sam", said)
        self.assertIn("Answered.", said)

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

        async def notice(text):
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

        async def notice(text):
            notices.append(text)

        agent._stream_once = slow_stream
        await agent.respond("chat", "take your time", on_notice=notice)

        self.assertGreaterEqual(len(notices), 1)
        self.assertEqual(set(notices), {"Still safely working."})

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
        first = agent._session_workdirs["chat"]
        self.assertTrue((first / "inputs").is_dir())
        self.assertTrue((first / "tmp").is_dir())
        self.assertTrue((first / "exports").is_dir())
        self.assertIn(str(first), self.sent["instructions"])
        self.assertIn(str(first / "exports"), self.sent["instructions"])

        await agent.forget("chat")
        await agent.respond("chat", "start over")
        second = agent._session_workdirs["chat"]
        self.assertNotEqual(first, second)
        self.assertEqual(first.name, "session-1")
        self.assertEqual(second.name, "session-2")

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

        session_root = agent._session_workdirs["chat"]
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


class OneShotTests(unittest.IsolatedAsyncioTestCase):
    """`pilotage ask` has to stay a one-shot.

    It is the check you run when the agent has gone quiet: does the login still
    work, does the model still answer. A check that remembers its own past
    questions answers differently every time it is run, and writes into the
    conversations of the agent it was meant to diagnose.
    """

    def test_a_store_without_a_path_keeps_nothing(self):
        store = ConversationStore(path=None)
        store.append("cli", [("user", "still there?"), ("assistant", "Yes.")])
        store.new_session("cli")
        self.assertEqual(store.load("cli", 10), [])

    async def test_the_ask_command_is_given_nowhere_to_write(self):
        stores = []

        class FakeAgent:
            def __init__(self, config, store=None):
                stores.append(store)

            async def respond(self, chat_id, question, on_notice=None):
                return "Answered."

            async def close(self):
                pass

        real = main.Agent
        main.Agent = FakeAgent
        self.addCleanup(setattr, main, "Agent", real)
        with redirect_stdout(StringIO()):
            self.assertEqual(await main.command_ask(Config.load(), "still there?"), 0)

        self.assertEqual(len(stores), 1)
        self.assertIsNotNone(stores[0], "ask fell back to the agent's own conversations")
        # Asked behaviourally: the store it was handed forgets what it is told.
        stores[0].append("cli", [("user", "still there?")])
        self.assertEqual(stores[0].load("cli", 10), [])


if __name__ == "__main__":
    unittest.main()
