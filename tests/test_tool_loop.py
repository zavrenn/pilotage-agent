"""A turn is many calls now.

The model asks for a tool, reads what came back, and goes again until it has
an answer. What is worth testing is not that a call happens, but that the
record of it is one the API will take back: reasoning, then a message, then
the calls, then their results, each output matched to its call by id — and
that a runaway loop still ends with the person getting what was found.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from pilotage.agent import MAX_ITERATIONS_SUMMARY_REQUEST, Agent
from pilotage.codex import stream as codex_stream
from pilotage.config import Config
from pilotage.history import ConversationStore


def _call(call_id: str, name: str = "todo", **arguments) -> Dict[str, str]:
    return {"call_id": call_id, "name": name, "arguments": json.dumps(arguments)}


class LoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.agent = Agent(Config.load(), ConversationStore(Path(tmp.name) / "conversations.db"))
        # Each element is what the model returns for the call at that position.
        self.replies: List[codex_stream.StreamResult] = []
        self.requests: List[Dict[str, Any]] = []

        async def _stream_once(request, *, force_refresh, ttfb_timeout, idle_timeout):
            self.requests.append(request)
            if self.replies:
                return self.replies.pop(0)
            return codex_stream.StreamResult(text="Done.")

        self.agent._stream_once = _stream_once

    def _sent(self, index: int = -1) -> List[Dict[str, Any]]:
        return self.requests[index]["input"]

    async def test_an_answer_with_no_tool_call_is_still_one_call(self):
        self.replies = [codex_stream.StreamResult(text="No tools needed.")]
        self.assertEqual(await self.agent.respond("chat", "hello"), "No tools needed.")
        self.assertEqual(len(self.requests), 1)

    async def test_a_tool_call_is_run_and_the_model_asked_again(self):
        self.replies = [
            codex_stream.StreamResult(
                tool_calls=[_call("call_1", todos=[{"id": "1", "content": "step", "status": "pending"}])]
            ),
            codex_stream.StreamResult(text="Planned."),
        ]
        self.assertEqual(await self.agent.respond("chat", "make a plan"), "Planned.")
        self.assertEqual(len(self.requests), 2)

    async def test_the_result_goes_back_matched_to_the_call_that_asked(self):
        self.replies = [
            codex_stream.StreamResult(tool_calls=[_call("call_abc")]),
            codex_stream.StreamResult(text="Read it."),
        ]
        await self.agent.respond("chat", "what is on the list?")
        outputs = [item for item in self._sent() if item.get("type") == "function_call_output"]
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["call_id"], "call_abc")
        self.assertIn("todos", json.loads(outputs[0]["output"]))

    async def test_the_call_is_replayed_before_its_result(self):
        self.replies = [
            codex_stream.StreamResult(text="Let me look.", tool_calls=[_call("call_abc")]),
            codex_stream.StreamResult(text="Read it."),
        ]
        await self.agent.respond("chat", "what is on the list?")
        types = [item.get("type") or item.get("role") for item in self._sent()]
        self.assertEqual(types[-3:], ["assistant", "function_call", "function_call_output"])

    async def test_reasoning_comes_before_the_message_it_belongs_to(self):
        """A reasoning item with no message after it is rejected as malformed."""
        self.replies = [
            codex_stream.StreamResult(
                reasoning_items=[{"type": "reasoning", "id": "rs_1", "encrypted_content": "x"}],
                tool_calls=[_call("call_abc")],
            ),
            codex_stream.StreamResult(text="Read it."),
        ]
        await self.agent.respond("chat", "think then look")
        types = [item.get("type") or item.get("role") for item in self._sent()]
        self.assertEqual(types[-4:], ["reasoning", "assistant", "function_call", "function_call_output"])

    async def test_a_replayed_reasoning_item_carries_no_server_id(self):
        """Nothing is stored server-side, so an id sent back is a 404."""
        self.replies = [
            codex_stream.StreamResult(
                reasoning_items=[
                    {"type": "reasoning", "id": "rs_1", "encrypted_content": "x", "_issuer_kind": "codex"}
                ],
                tool_calls=[_call("call_abc")],
            ),
            codex_stream.StreamResult(text="Read it."),
        ]
        await self.agent.respond("chat", "think then look")
        reasoning = [item for item in self._sent() if item.get("type") == "reasoning"]
        self.assertNotIn("id", reasoning[0])
        self.assertNotIn("_issuer_kind", reasoning[0])

    async def test_several_calls_in_one_step_all_come_back(self):
        self.replies = [
            codex_stream.StreamResult(tool_calls=[_call("call_1"), _call("call_2")]),
            codex_stream.StreamResult(text="Both done."),
        ]
        await self.agent.respond("chat", "two things")
        outputs = [item for item in self._sent() if item.get("type") == "function_call_output"]
        self.assertEqual([item["call_id"] for item in outputs], ["call_1", "call_2"])

    async def test_a_tool_that_fails_is_reported_to_the_model_not_to_the_person(self):
        self.replies = [
            codex_stream.StreamResult(tool_calls=[_call("call_1", name="not_a_tool")]),
            codex_stream.StreamResult(text="I will do it another way."),
        ]
        answer = await self.agent.respond("chat", "do the impossible")
        self.assertEqual(answer, "I will do it another way.")
        outputs = [item for item in self._sent() if item.get("type") == "function_call_output"]
        self.assertIn("error", json.loads(outputs[0]["output"]))

    async def test_the_work_of_a_turn_is_remembered_for_the_next_one(self):
        self.replies = [
            codex_stream.StreamResult(tool_calls=[_call("call_1")]),
            codex_stream.StreamResult(text="Read it."),
        ]
        await self.agent.respond("chat", "what is on the list?")
        await self.agent.respond("chat", "and now?")
        types = [item.get("type") for item in self._sent()]
        self.assertIn("function_call", types)
        self.assertIn("function_call_output", types)

    async def test_the_task_list_survives_across_turns_of_one_chat(self):
        self.replies = [
            codex_stream.StreamResult(
                tool_calls=[_call("call_1", todos=[{"id": "1", "content": "step", "status": "pending"}])]
            ),
            codex_stream.StreamResult(text="Planned."),
        ]
        await self.agent.respond("chat", "make a plan")

        self.replies = [
            codex_stream.StreamResult(tool_calls=[_call("call_2")]),
            codex_stream.StreamResult(text="Still one step."),
        ]
        await self.agent.respond("chat", "what was the plan?")
        outputs = [item for item in self._sent() if item.get("type") == "function_call_output"]
        self.assertEqual(len(json.loads(outputs[-1]["output"])["todos"]), 1)

    async def test_ending_the_conversation_ends_its_task_list_too(self):
        self.replies = [
            codex_stream.StreamResult(
                tool_calls=[_call("call_1", todos=[{"id": "1", "content": "step", "status": "pending"}])]
            ),
            codex_stream.StreamResult(text="Planned."),
        ]
        await self.agent.respond("chat", "make a plan")
        await self.agent.forget("chat")

        self.replies = [
            codex_stream.StreamResult(tool_calls=[_call("call_2")]),
            codex_stream.StreamResult(text="Nothing here."),
        ]
        await self.agent.respond("chat", "what was the plan?")
        outputs = [item for item in self._sent() if item.get("type") == "function_call_output"]
        self.assertEqual(json.loads(outputs[-1]["output"])["todos"], [])

    async def test_chats_do_not_share_a_task_list(self):
        self.replies = [
            codex_stream.StreamResult(
                tool_calls=[_call("call_1", todos=[{"id": "1", "content": "step", "status": "pending"}])]
            ),
            codex_stream.StreamResult(text="Planned."),
        ]
        await self.agent.respond("mine", "make a plan")

        self.replies = [
            codex_stream.StreamResult(tool_calls=[_call("call_2")]),
            codex_stream.StreamResult(text="Nothing here."),
        ]
        await self.agent.respond("theirs", "what is on my list?")
        outputs = [item for item in self._sent() if item.get("type") == "function_call_output"]
        self.assertEqual(json.loads(outputs[-1]["output"])["todos"], [])


class ToolOfferTests(unittest.IsolatedAsyncioTestCase):
    """What the model is told it can do."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.requests: List[Dict[str, Any]] = []

    def _agent(self, config: Config) -> Agent:
        agent = Agent(config, ConversationStore(self.tmp / "conversations.db"))

        async def _stream_once(request, *, force_refresh, ttfb_timeout, idle_timeout):
            self.requests.append(request)
            return codex_stream.StreamResult(text="Done.")

        agent._stream_once = _stream_once
        return agent

    async def test_the_enabled_tools_are_sent_with_the_request(self):
        await self._agent(Config.load()).respond("chat", "hello")
        names = [tool["name"] for tool in self.requests[-1]["tools"]]
        self.assertIn("todo", names)
        self.assertEqual(self.requests[-1]["tool_choice"], "auto")

    async def test_an_agent_with_nothing_enabled_sends_no_tool_key_at_all(self):
        """The SDK iterates `tools` without a None guard; an empty list breaks it."""
        (self.tmp / "config.yaml").write_text("tools:\n  enabled: []\n", encoding="utf-8")
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"PILOTAGE_CONFIG": str(self.tmp / "config.yaml")}):
            agent = self._agent(Config.load())
        await agent.respond("chat", "hello")
        self.assertNotIn("tools", self.requests[-1])

    async def test_the_tool_list_does_not_change_under_a_conversation(self):
        agent = self._agent(Config.load())
        await agent.respond("chat", "one")
        await agent.respond("chat", "two")
        self.assertEqual(self.requests[0]["tools"], self.requests[1]["tools"])
        self.assertEqual(
            self.requests[0]["prompt_cache_key"], self.requests[1]["prompt_cache_key"]
        )


class RunawayTests(unittest.IsolatedAsyncioTestCase):
    """A model that never stops asking still has to produce an answer."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        import os
        from unittest import mock

        config_file = Path(tmp.name) / "config.yaml"
        config_file.write_text("tools:\n  max_iterations: 3\n", encoding="utf-8")
        patch = mock.patch.dict(os.environ, {"PILOTAGE_CONFIG": str(config_file)})
        patch.start()
        self.addCleanup(patch.stop)

        self.agent = Agent(Config.load(), ConversationStore(Path(tmp.name) / "conversations.db"))
        self.requests: List[Dict[str, Any]] = []

        async def _stream_once(request, *, force_refresh, ttfb_timeout, idle_timeout):
            self.requests.append(request)
            if "tools" not in request:
                return codex_stream.StreamResult(text="Here is what I found.")
            return codex_stream.StreamResult(tool_calls=[_call(f"call_{len(self.requests)}")])

        self.agent._stream_once = _stream_once

    async def test_the_loop_stops_at_the_configured_limit(self):
        await self.agent.respond("chat", "go forever")
        # Three steps, then one last call asking for a summary.
        self.assertEqual(len(self.requests), 4)

    async def test_the_person_waiting_gets_what_was_found(self):
        with self.assertLogs("pilotage.agent", level="WARNING"):
            answer = await self.agent.respond("chat", "go forever")
        self.assertEqual(answer, "Here is what I found.")

    async def test_the_last_call_offers_no_tools(self):
        await self.agent.respond("chat", "go forever")
        self.assertNotIn("tools", self.requests[-1])
        said = [item.get("content") for item in self.requests[-1]["input"]]
        self.assertIn(MAX_ITERATIONS_SUMMARY_REQUEST, said)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
