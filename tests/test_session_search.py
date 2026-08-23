"""Hermes-style recall over Genesis' own durable conversation store."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pilotage.agent import Agent
from pilotage.codex import stream as codex_stream
from pilotage.config import Config
from pilotage.history import ConversationStore, SCHEMA
from pilotage.tools import ToolContext, build_registry
from pilotage.tools.session_search import sanitize_fts5_query


class SessionSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = ConversationStore(self.root / "conversations.db")
        self.registry = build_registry()
        self.context = ToolContext(
            chat_id="current",
            config=None,
            conversation_store=self.store,
        )

    def seed(self, chat_id: str, messages):
        self.store.append(chat_id, messages)

    async def call(self, **arguments):
        raw = await self.registry.dispatch(
            "session_search",
            json.dumps(arguments),
            self.context,
            allowed_groups=["session_search"],
        )
        return json.loads(raw)

    async def test_discovery_finds_a_prior_session_but_not_active_context(self):
        self.seed(
            "current",
            [
                ("user", "The moonstone deployment uses port 7443."),
                ("assistant", "I recorded the moonstone deployment."),
            ],
        )
        self.store.new_session("current")
        self.seed(
            "current",
            [
                ("user", "The active moonstone draft is already in context."),
                ("assistant", "Continuing the active draft."),
            ],
        )

        result = await self.call(query="moonstone")

        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "discover")
        self.assertEqual(result["count"], 1)
        blob = json.dumps(result, ensure_ascii=False)
        self.assertIn("port 7443", blob)
        self.assertNotIn("active moonstone draft", blob)

    async def test_search_is_profile_wide_like_hermes(self):
        self.seed(
            "another-chat",
            [
                ("user", "The cedar invoice review is complete."),
                ("assistant", "Cedar was reconciled."),
            ],
        )

        result = await self.call(query="cedar")

        self.assertEqual(result["count"], 1)
        self.assertIn("cedar", json.dumps(result).lower())

    async def test_an_explicit_store_keeps_profiles_separate(self):
        other = ConversationStore(self.root / "other-profile.db")
        other.append(
            "another-chat",
            [("user", "Private obsidian profile note.")],
        )

        result = await self.call(query="obsidian")

        self.assertEqual(result["count"], 0)

    async def test_adaptive_detail_hydrates_only_the_top_result(self):
        for chat in ("one", "two"):
            self.seed(
                chat,
                [
                    ("user", f"Start {chat} about the quartz rollout."),
                    ("assistant", f"Finish {chat} quartz rollout."),
                ],
            )

        result = await self.call(query="quartz", limit=2)

        self.assertEqual(result["count"], 2)
        self.assertEqual(
            [item["detail"] for item in result["results"]],
            ["full", "compact"],
        )
        self.assertTrue(result["results"][0]["bookend_start"])
        self.assertEqual(result["results"][1]["bookend_start"], [])
        self.assertEqual(len(result["results"][1]["messages"]), 1)

    async def test_full_detail_hydrates_every_result(self):
        for chat in ("one", "two"):
            self.seed(chat, [("user", f"{chat} amber roadmap")])

        result = await self.call(query="amber", limit=2, detail="full")

        self.assertEqual(result["count"], 2)
        self.assertEqual(
            [item["detail"] for item in result["results"]],
            ["full", "full"],
        )
        self.assertTrue(all(item["bookend_start"] for item in result["results"]))

    async def test_browse_lists_recent_history_without_active_session(self):
        self.seed("current", [("user", "older current session")])
        self.store.new_session("current")
        self.seed("current", [("user", "active session must stay out")])
        self.seed("other", [("user", "other stored session")])

        result = await self.call(limit=10)

        self.assertEqual(result["mode"], "browse")
        self.assertEqual(result["count"], 2)
        blob = json.dumps(result)
        self.assertIn("older current session", blob)
        self.assertIn("other stored session", blob)
        self.assertNotIn("active session must stay out", blob)

    async def test_read_returns_a_small_session_in_full(self):
        self.seed(
            "old",
            [("user", "start silver"), ("assistant", "finish silver")],
        )
        discovered = await self.call(query="silver")
        session_id = discovered["results"][0]["session_id"]

        result = await self.call(session_id=session_id)

        self.assertEqual(result["mode"], "read")
        self.assertFalse(result["truncated"])
        self.assertEqual(result["message_count"], 2)
        self.assertEqual(len(result["messages"]), 2)

    async def test_read_bounds_a_large_session_to_hermes_head_and_tail(self):
        messages = [
            (
                "user" if index % 2 == 0 else "assistant",
                f"long session message {index}",
            )
            for index in range(36)
        ]
        self.seed("old", messages)
        browsed = await self.call(limit=10)
        session_id = browsed["results"][0]["session_id"]

        result = await self.call(session_id=session_id)

        self.assertTrue(result["truncated"])
        self.assertEqual(result["message_count"], 36)
        self.assertEqual(len(result["messages"]), 30)
        self.assertEqual(result["messages"][0]["content"], "long session message 0")
        self.assertEqual(result["messages"][-1]["content"], "long session message 35")

    async def test_scroll_centers_a_bounded_window_on_the_anchor(self):
        self.seed(
            "old",
            [
                (
                    "user" if index % 2 == 0 else "assistant",
                    f"scroll message {index}",
                )
                for index in range(9)
            ],
        )
        read = await self.call(
            session_id=(await self.call(limit=10))["results"][0]["session_id"]
        )
        session_id = read["session_id"]
        anchor = read["messages"][4]["id"]

        result = await self.call(
            session_id=session_id,
            around_message_id=anchor,
            window=2,
        )

        self.assertEqual(result["mode"], "scroll")
        self.assertEqual(len(result["messages"]), 5)
        self.assertEqual(result["messages_before"], 2)
        self.assertEqual(result["messages_after"], 2)
        matches = [message for message in result["messages"] if message.get("match")]
        self.assertEqual([message["id"] for message in matches], [anchor])

    async def test_role_filter_limits_discovery(self):
        self.seed(
            "old",
            [
                ("user", "user says cobalt"),
                ("assistant", "assistant says vermilion"),
            ],
        )

        absent = await self.call(query="vermilion", role_filter="user")
        present = await self.call(query="vermilion", role_filter="assistant")

        self.assertEqual(absent["count"], 0)
        self.assertEqual(present["count"], 1)

    async def test_invalid_role_filter_is_a_visible_tool_error(self):
        result = await self.call(query="anything", role_filter="tool")
        self.assertFalse(result["success"])
        self.assertIn("unsupported", result["error"])

    async def test_punctuation_query_is_sanitized_instead_of_breaking_fts(self):
        self.seed(
            "old",
            [("user", "Inspect gateway/run.py before deployment.")],
        )

        result = await self.call(query="gateway/run.py")

        self.assertEqual(result["count"], 1)
        self.assertEqual(
            sanitize_fts5_query("gateway/run.py"),
            'gateway "run.py"',
        )

    async def test_malformed_query_becomes_an_empty_safe_search(self):
        result = await self.call(query='" OR ()')
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 0)

    async def test_boolean_or_query_keeps_hermes_fts_syntax(self):
        self.seed("one", [("user", "alpha constellation")])
        self.seed("two", [("user", "beta constellation")])

        result = await self.call(query="alpha OR beta", limit=10)

        self.assertEqual(result["count"], 2)

    async def test_interactive_history_ranks_ahead_of_repetitive_cron_sessions(self):
        for index in range(12):
            self.seed(
                f"cron:job-{index}:owner",
                [
                    (
                        "assistant",
                        "opal forecast " * 20 + f"scheduled result {index}",
                    )
                ],
            )
        self.seed("person", [("user", "We chose the opal forecast manually.")])

        result = await self.call(query="opal forecast", limit=3)

        self.assertIn("manually", json.dumps(result["results"][0]))

    async def test_discovery_caps_windows_and_bookends(self):
        self.seed(
            "old",
            [
                ("user", "needle " + "x" * 6_000),
                ("assistant", "done"),
            ],
        )

        result = await self.call(query="needle")
        first = result["results"][0]

        self.assertTrue(
            all(len(message["content"]) <= 4_000 for message in first["messages"])
        )
        self.assertTrue(
            all(
                len(message["content"]) <= 1_200
                for message in first["bookend_start"] + first["bookend_end"]
            )
        )

    async def test_missing_session_and_bad_scroll_are_visible_errors(self):
        missing = await self.call(session_id="999999")
        no_session = await self.call(around_message_id=1)

        self.assertFalse(missing["success"])
        self.assertFalse(no_session["success"])

    async def test_one_shot_store_reports_search_unavailable(self):
        context = ToolContext(
            chat_id="cli",
            config=None,
            conversation_store=ConversationStore(path=None),
        )
        raw = await self.registry.dispatch(
            "session_search",
            json.dumps({"query": "anything"}),
            context,
            allowed_groups=["session_search"],
        )
        result = json.loads(raw)

        self.assertFalse(result["success"])
        self.assertIn("one-shot", result["error"])


class SessionSearchMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_turns_are_backfilled_when_fts_is_added(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "legacy.db"
        connection = sqlite3.connect(path)
        try:
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT INTO turns"
                " (chat_id, session, role, content, replay, written_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                ("old", 1, "user", "legacy sapphire decision", "", 1.0),
            )
            connection.commit()
        finally:
            connection.close()

        registry = build_registry()
        raw = await registry.dispatch(
            "session_search",
            json.dumps({"query": "sapphire"}),
            ToolContext(
                chat_id="current",
                config=None,
                conversation_store=ConversationStore(path),
            ),
            allowed_groups=["session_search"],
        )
        result = json.loads(raw)

        self.assertEqual(result["count"], 1)
        self.assertIn("sapphire", json.dumps(result).lower())


class AgentWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_supplies_its_own_store_to_session_search(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = ConversationStore(Path(temporary.name) / "conversations.db")
        store.append("chat", [("user", "historic topaz choice")])
        store.new_session("chat")
        agent = Agent(Config.load(), store)
        replies = [
            codex_stream.StreamResult(
                tool_calls=[
                    {
                        "call_id": "call_search",
                        "name": "session_search",
                        "arguments": json.dumps({"query": "topaz"}),
                    }
                ]
            ),
            codex_stream.StreamResult(text="Found it."),
        ]
        requests = []

        async def stream_once(request, **_kwargs):
            requests.append(request)
            return replies.pop(0)

        agent._stream_once = stream_once

        self.assertEqual(await agent.respond("chat", "What did we choose?"), "Found it.")
        outputs = [
            item
            for item in requests[-1]["input"]
            if item.get("type") == "function_call_output"
        ]
        result = json.loads(outputs[0]["output"])
        self.assertEqual(result["count"], 1)
        self.assertIn("topaz", json.dumps(result).lower())


if __name__ == "__main__":
    unittest.main()
