"""Conversations that outlive the process.

A restart is invisible from a phone. The person writes again, and the agent
either remembers or does not — so the thing worth testing is not that rows are
written, it is that a second Agent over the same file picks the conversation up
where the first one left it.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from io import StringIO
from pathlib import Path

from pilotage import main
from pilotage.agent import Agent
from pilotage.codex import stream as codex_stream
from pilotage.config import Config
from pilotage.history import ConversationStore


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


    def test_a_broken_file_costs_a_log_line_not_the_turn(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"this is not a database")
        with self.assertLogs("pilotage.history", level="WARNING"):
            self.store.append("chat", [("user", "hello")])
        with self.assertLogs("pilotage.history", level="WARNING"):
            self.assertEqual(self.store.load("chat", 10), [])


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

    async def test_a_store_that_cannot_be_written_still_answers(self):
        broken = Agent(Config.load(), ConversationStore(self.path))
        self.path.write_bytes(b"this is not a database")

        async def _stream_once(request, *, force_refresh, ttfb_timeout, idle_timeout):
            return codex_stream.StreamResult(text="Answered anyway.")

        broken._stream_once = _stream_once
        with self.assertLogs("pilotage.history", level="WARNING"):
            answer = await broken.respond("chat", "hello")
        self.assertEqual(answer, "Answered anyway.")


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
