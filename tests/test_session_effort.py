"""Session effort isolation, durability, and exact-turn recovery."""

import asyncio
import os
import sqlite3
import tempfile
import threading
import unittest
from dataclasses import replace
from contextlib import closing
from pathlib import Path
from unittest import mock

from pilotage.agent import Agent
from pilotage.codex import auth, models
from pilotage.codex.stream import StreamResult
from pilotage.commands import execute_command, parse_command
from pilotage.config import Config
from pilotage.history import ConversationError, ConversationStore
from pilotage.i18n import t


class EffortTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "config.yaml").write_text("tools:\n  enabled: []\ndisplay:\n  language: en\n", encoding="utf-8")
        environment = {"PILOTAGE_HOME": str(self.root)}
        # Windows' SSL provider needs SystemRoot when Config first imports the
        # Telegram dependency; this test module must also work on its own.
        if "SystemRoot" in os.environ:
            environment["SystemRoot"] = os.environ["SystemRoot"]
        with mock.patch.dict(os.environ, environment, clear=True):
            self.config = Config.load()
        self.config = replace(self.config, working_notice_interval_seconds=0)
        self.agent = self.new_agent()
        catalog = mock.patch.object(models, "available_efforts", return_value=models.EFFORTS)
        self.catalog = catalog.start()
        self.addCleanup(catalog.stop)

    def new_agent(self, config=None, **kwargs):
        agent = Agent(config or self.config, **kwargs)
        self.addAsyncCleanup(agent.close)
        return agent

    async def command(self, text, *, chat="a", agent=None):
        agent = agent or self.agent
        return await execute_command(
            parse_command(text), agent=agent, config=agent._config,
            session_id=chat, reset_reply="reset",
        )

    async def test_check_is_read_only_and_leaves_due_resets_for_the_next_turn(self):
        for mode in ("idle", "daily"):
            with self.subTest(mode=mode):
                timed = self.new_agent(replace(self.config, session_reset_mode=mode, session_reset_idle_minutes=1))
                store = timed._store
                previous = [("user", "previous context"), ("assistant", "remembered")]
                store.append(mode, previous)
                store.session_effort(mode, "max")
                with store._connect() as db:
                    db.execute("UPDATE chats SET last_active = 1 WHERE chat_id = ?", (mode,))
                generation = store.current_session(mode)
                await timed._restore(mode)
                live_history = list(timed._history[mode])

                reply = await self.command("/effort", chat=mode, agent=timed)

                self.assertIn("Session effort: max", reply)
                self.assertEqual(store.current_session(mode), generation)
                self.assertEqual(store.load(mode, 10), previous)
                self.assertEqual(timed._history[mode], live_history)
                with store._connect() as db:
                    self.assertEqual(db.execute("SELECT last_active FROM chats WHERE chat_id = ?", (mode,)).fetchone()[0], 1)

                notices = []
                async def notice(text, _replace_id=""):
                    notices.append(text)
                with mock.patch.object(timed, "_stream_once", return_value=StreamResult(text="Done", terminal_completed=True)) as stream:
                    await timed.respond(mode, "next message", on_notice=notice)
                request = stream.call_args.args[0]
                self.assertEqual(request["reasoning"]["effort"], "high")
                self.assertEqual(store.current_session(mode), generation + 1)
                self.assertEqual(len(notices), 1)
                self.assertIn("fresh conversation", str(request["input"]))
                self.assertNotIn("previous context", str(request["input"]))

    async def test_chat_options_and_validation_use_this_accounts_catalog(self):
        self.catalog.return_value = ("low", "high")
        self.assertIn("Available: low, high.", await self.command("/effort"))
        for value in ("max", "ultra", "unknown"):
            reply = await self.command("/effort " + value)
            self.assertIn("unavailable", reply)
            self.assertIn("Available: low, high.", reply)
            self.assertEqual(await self.agent.session_effort("a"), "high")
        self.assertIn("set to low", await self.command("/effort low"))
        with mock.patch.object(self.agent, "_stream_once", return_value=StreamResult(text="Done", terminal_completed=True)) as stream:
            await self.agent.respond("a", "next message")
        self.assertEqual(stream.call_args.args[0]["reasoning"]["effort"], "low")
        self.assertTrue(self.catalog.call_args_list)
        self.assertTrue(all(call == mock.call(self.config.credentials_path) for call in self.catalog.call_args_list))

    async def test_check_does_not_create_a_chat_and_only_suggests_an_available_level(self):
        self.catalog.return_value = ("low",)
        reply = await self.command("/effort")
        self.assertIn("Session effort: high", reply)
        self.assertIn("Available: low.", reply)
        self.assertIn("/effort low", reply)
        self.assertNotIn("/effort high", reply)
        with self.agent._store._connect() as db:
            self.assertIsNone(db.execute("SELECT 1 FROM chats WHERE chat_id = 'a'").fetchone())

    async def test_catalog_failure_keeps_checks_readable_and_rejects_changes(self):
        self.agent._store.session_effort("a", "max")
        for error in (models.ModelError("private detail"), auth.AuthError("private detail"), OSError("private detail")):
            with self.subTest(error=type(error).__name__):
                self.catalog.side_effect = error
                reply = await self.command("/effort")
                self.assertIn("Session effort: max", reply)
                self.assertIn("couldn't check", reply)
                self.assertNotIn("private detail", reply)
                reply = await self.command("/effort low")
                self.assertIn("No changes were made", reply)
                self.assertNotIn("private detail", reply)
                self.assertEqual(await self.agent.session_effort("a"), "max")

    async def test_rejected_choices_leave_an_idle_session_untouched(self):
        timed = self.new_agent(replace(self.config, session_reset_mode="idle", session_reset_idle_minutes=1))
        store = timed._store
        previous = [("user", "previous context"), ("assistant", "remembered")]
        store.append("a", previous)
        store.session_effort("a", "max")
        with store._connect() as db:
            db.execute("UPDATE chats SET last_active = 1 WHERE chat_id = 'a'")
        generation = store.current_session("a")
        self.catalog.return_value = ("low", "high")
        self.assertIn("unavailable", await self.command("/effort max", agent=timed))
        self.catalog.side_effect = models.ModelError("unavailable")
        self.assertIn("No changes were made", await self.command("/effort low", agent=timed))
        with self.assertRaises(ValueError):
            await timed.session_effort("a", "ultra")
        self.assertEqual(store.current_session("a"), generation)
        self.assertEqual(store.load("a", 10), previous)
        self.assertEqual(store.session_effort("a"), "max")
        with store._connect() as db:
            self.assertEqual(db.execute("SELECT last_active FROM chats WHERE chat_id = 'a'").fetchone()[0], 1)

    async def test_choice_after_a_due_reset_includes_the_configured_notice(self):
        for notify in (True, False):
            with self.subTest(notify=notify):
                chat = str(notify)
                timed = self.new_agent(replace(self.config, session_reset_mode="idle", session_reset_idle_minutes=1, session_reset_notify=notify))
                store = timed._store
                store.append(chat, [("user", "previous context")])
                store.session_effort(chat, "max")
                with store._connect() as db:
                    db.execute("UPDATE chats SET last_active = 1 WHERE chat_id = ?", (chat,))
                generation = store.current_session(chat)

                reply = await self.command("/effort low", chat=chat, agent=timed)

                self.assertEqual(t("session.auto_reset_idle", "en") in reply, notify)
                self.assertIn("set to low", reply)
                self.assertEqual(store.current_session(chat), generation + 1)
                self.assertEqual(await timed.session_effort(chat), "low")

    async def test_reset_notice_is_not_lost_if_the_effort_save_fails(self):
        timed = self.new_agent(replace(self.config, session_reset_mode="idle", session_reset_idle_minutes=1))
        store = timed._store
        store.append("a", [("user", "previous context")])
        with store._connect() as db:
            db.execute("UPDATE chats SET last_active = 1 WHERE chat_id = 'a'")
        with mock.patch.object(store, "session_effort", side_effect=ConversationError("save failed")):
            reply = await self.command("/effort max", agent=timed)
        self.assertIn(t("session.auto_reset_idle", "en"), reply)
        self.assertIn("couldn't read or save", reply)
        self.assertNotIn("set to max", reply)

    async def test_cancelled_catalog_lookup_cannot_save_a_late_choice(self):
        started = asyncio.Event()
        finished = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def delayed(path):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(3):
                raise AssertionError("catalog lookup was not released")
            loop.call_soon_threadsafe(finished.set)
            return models.EFFORTS

        self.catalog.side_effect = delayed
        choosing = asyncio.create_task(self.command("/effort max"))
        try:
            await asyncio.wait_for(started.wait(), 1)
            choosing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(choosing, 1)
        finally:
            release.set()
        await asyncio.wait_for(finished.wait(), 1)
        self.assertEqual(await self.agent.session_effort("a"), "high")

    async def test_two_command_forms_validate_and_isolate_sessions(self):
        self.assertIn("Session effort: high", await self.command("/effort"))
        self.assertIn("set to max", await self.command("/effort max"))
        self.assertEqual(await self.agent.session_effort("a"), "max")
        self.assertEqual(await self.agent.session_effort("b"), "high")
        for value in ("default", "none", "ultra", "high extra", "low; /new"):
            self.assertIn("Available:", await self.command("/effort " + value))
            self.assertEqual(await self.agent.session_effort("a"), "max")
        self.assertEqual(self.agent._store.load("a", 10), [])

    async def test_restart_preserves_choice_and_new_resets_only_that_session(self):
        await self.command("/effort max")
        await self.command("/effort low", chat="b")
        restarted = self.new_agent()
        self.assertEqual(await restarted.session_effort("a"), "max")
        self.assertEqual(await self.command("/new", agent=restarted), "reset")
        self.assertEqual(await self.new_agent().session_effort("a"), "high")
        self.assertEqual(await restarted.session_effort("b"), "low")

    async def test_failed_save_never_confirms_a_change(self):
        with mock.patch.object(self.agent._store, "session_effort", side_effect=ConversationError("disk failed")):
            answer = await self.command("/effort max")
        self.assertIn("couldn't read or save", answer)
        self.assertEqual(await self.agent.session_effort("a"), "high")

    async def test_failed_new_keeps_the_effort(self):
        await self.command("/effort max")
        with mock.patch.object(self.agent._store, "new_session", side_effect=ConversationError("failed")), self.assertRaises(ConversationError):
            await self.agent.forget("a")
        self.assertEqual(await self.agent.session_effort("a"), "max")

    async def test_active_turn_keeps_its_effort_across_continuations(self):
        started = asyncio.Event()
        release = asyncio.Event()
        requests = []

        async def stream(request, **kwargs):
            requests.append(request)
            if len(requests) == 1:
                started.set()
                await release.wait()
                return StreamResult(text="Working", needs_continuation=True)
            return StreamResult(text="Done", terminal_completed=True)

        with mock.patch.object(self.agent, "_stream_once", side_effect=stream):
            running = asyncio.create_task(self.agent.respond("a", "do work"))
            try:
                await asyncio.wait_for(started.wait(), 2)
                await asyncio.wait_for(self.command("/effort max"), 2)
                self.assertFalse(running.done())
                active = self.agent._store.list_active_turns()[0]
                self.assertEqual(active.reasoning_effort, "high")
            finally:
                release.set()
                await running
            await self.agent.respond("a", "next task")
        self.assertEqual([r["reasoning"]["effort"] for r in requests], ["high", "high", "max"])
        self.assertTrue(all(r["model"] == models.MODEL for r in requests))
        self.assertTrue(all(r["context_management"] for r in requests))

    async def test_recovery_uses_original_effort_and_keeps_compaction(self):
        store = self.agent._store
        store.append_with_replay("a", [("user", "before", []), ("assistant", "saved", [{"type": "compaction", "encrypted_content": "checkpoint"}])])
        store.begin_turn("a", "unfinished", reasoning_effort="low")
        active = store.list_active_turns()[0]
        await self.command("/effort max")
        restarted = self.new_agent(replace(self.config, reasoning_effort="high"))
        with mock.patch.object(restarted, "_stream_once", return_value=StreamResult(text="Done", terminal_completed=True)) as stream:
            await restarted.recover_turn(active)
        request = stream.call_args.args[0]
        self.assertEqual(request["reasoning"]["effort"], "low")
        self.assertEqual(request["input"][0]["encrypted_content"], "checkpoint")
        self.assertEqual(await restarted.session_effort("a"), "max")
        self.assertEqual(store.list_active_turns(), [])

    async def test_request_effort_is_frozen_before_voice_or_attachment_preparation(self):
        async with self.agent.prepare_turn("a") as execution:
            await self.command("/effort max")
            with mock.patch.object(self.agent, "_stream_once", return_value=StreamResult(text="Done", terminal_completed=True)) as stream:
                await self.agent.respond_result("a", "transcribed message", prepared_execution=execution)
        self.assertEqual(stream.call_args.args[0]["reasoning"]["effort"], "high")
        self.assertEqual(await self.agent.session_effort("a"), "max")

    async def test_cancelled_save_cannot_overwrite_a_later_new_session(self):
        saving = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        original = self.agent._store.session_effort

        def delayed(chat, value=None):
            loop.call_soon_threadsafe(saving.set)
            if not release.wait(3):
                raise AssertionError("save was never released")
            return original(chat, value)

        with mock.patch.object(self.agent._store, "session_effort", side_effect=delayed):
            setting = asyncio.create_task(self.agent.session_effort("a", "max"))
            try:
                await asyncio.wait_for(saving.wait(), 1)
                setting.cancel()
                resetting = asyncio.create_task(self.agent.forget("a"))
                await asyncio.sleep(0)
                self.assertFalse(resetting.done())
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await setting
            self.assertTrue(await resetting)
        self.assertEqual(await self.agent.session_effort("a"), "high")

    async def test_automatic_reset_clears_effort_and_new_choice_targets_new_session(self):
        await self.command("/effort high")
        old = self.agent._store.current_session("a")
        with self.agent._store._connect() as db:
            db.execute("UPDATE chats SET last_active = 1 WHERE chat_id = 'a'")
        timed = self.new_agent(replace(self.config, session_reset_mode="idle", session_reset_idle_minutes=1))
        await timed.session_effort("a", "max")
        self.assertEqual(timed._store.current_session("a"), old + 1)
        self.assertEqual(await timed.session_effort("a"), "max")
        self.assertEqual(await timed.session_effort("b"), "high")

    async def test_session_effort_does_not_change_scheduled_work(self):
        await self.command("/effort max")
        scheduled = self.new_agent(store=ConversationStore(None), scheduled_run=True)
        with mock.patch.object(scheduled, "_stream_once", return_value=StreamResult(text="Done", terminal_completed=True)) as stream:
            await scheduled.respond("a", "scheduled task")
        self.assertEqual(stream.call_args.args[0]["reasoning"]["effort"], "high")

    async def test_corrupt_effort_stops_before_the_model(self):
        await self.command("/effort high")
        with self.agent._store._connect() as db:
            db.execute("UPDATE chats SET reasoning_effort = 'ultra' WHERE chat_id = 'a'")
        with mock.patch.object(self.agent, "_stream_once") as stream, self.assertRaises(ConversationError):
            await self.agent.respond("a", "hello")
        stream.assert_not_called()


class ExistingSchemaTests(unittest.TestCase):
    def test_existing_database_can_store_efforts_without_losing_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.db"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("CREATE TABLE chats (chat_id TEXT PRIMARY KEY, session INTEGER NOT NULL, last_active REAL NOT NULL DEFAULT 0)")
                db.execute("INSERT INTO chats VALUES ('a', 3, 1)")
            store = ConversationStore(path)
            store.append("a", [("user", "keep this")])
            store.session_effort("a", "high")
            self.assertEqual(store.current_session("a"), 3)
            self.assertEqual(store.load("a", 10), [("user", "keep this")])
            self.assertEqual(ConversationStore(path).session_effort("a"), "high")
