"""A turn is many calls now.

The model asks for a tool, reads what came back, and goes again until it has
an answer. What is worth testing is not that a call happens, but that the
record of it is one the API will take back: reasoning, then a message, then
the calls, then their results, each output matched to its call by id — and
that a runaway loop still ends with the person getting what was found.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from pilotage import media
from pilotage.agent import (
    CODEX_INCOMPLETE_RESPONSE,
    MAX_ITERATIONS_SUMMARY_REQUEST,
    Agent,
)
from pilotage.codex import stream as codex_stream
from pilotage.config import Config
from pilotage.history import ConversationError, ConversationStore
from pilotage.tools import Tool


def _call(call_id: str, name: str = "todo", **arguments) -> Dict[str, str]:
    return {"call_id": call_id, "name": name, "arguments": json.dumps(arguments)}


def _message(
    text: str,
    *,
    phase: str,
    item_id: str = "msg_1",
    status: str = "completed",
) -> Dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "status": status,
        "id": item_id,
        "phase": phase,
        "content": [{"type": "output_text", "text": text}],
    }


class LoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.agent = Agent(Config.load(), ConversationStore(self.root / "conversations.db"))
        # Keep staged attachment fixtures inside this test's temporary state;
        # the operator's default profile workspace may contain real inputs.
        self.agent._context_cwd = self.root / "workspace"
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

    async def test_persisted_turn_exposes_positive_terminal_proof(self):
        self.replies = [
            codex_stream.StreamResult(
                text="Finished.", terminal_completed=True
            )
        ]

        result = await self.agent.respond_result("chat", "finish")

        self.assertTrue(result.terminal_completed)
        self.assertEqual(
            self.agent._store.load("chat", 2)[-1],
            ("assistant", "Finished."),
        )

    async def test_nonterminal_tool_call_is_not_executed(self):
        self.replies = [
            codex_stream.StreamResult(
                text="partial",
                tool_calls=[
                    _call(
                        "call_1",
                        todos=[{"id": "1", "content": "must not run"}],
                    )
                ],
                terminal_completed=False,
            )
        ]

        result = await self.agent.respond_result("chat", "do it")

        self.assertFalse(result.terminal_completed)
        self.assertNotIn("todo", self.agent._tool_state["chat"])
        self.assertEqual(len(self.requests), 1)

    async def test_native_compaction_clears_stale_skill_view_dedup(self):
        self.agent._tool_state["chat"] = {
            "skill_views": {("report", ""): (1, 1)}
        }
        self.replies = [
            codex_stream.StreamResult(
                text="Compacted.",
                reasoning_items=[
                    {"type": "compaction", "encrypted_content": "opaque"}
                ],
            )
        ]
        self.assertEqual(await self.agent.respond("chat", "continue"), "Compacted.")
        self.assertNotIn("skill_views", self.agent._tool_state["chat"])

    async def test_a_tool_call_is_run_and_the_model_asked_again(self):
        self.replies = [
            codex_stream.StreamResult(
                tool_calls=[_call("call_1", todos=[{"id": "1", "content": "step", "status": "pending"}])]
            ),
            codex_stream.StreamResult(text="Planned."),
        ]
        self.assertEqual(await self.agent.respond("chat", "make a plan"), "Planned.")
        self.assertEqual(len(self.requests), 2)

    async def test_tool_work_does_not_run_if_its_call_cannot_be_persisted(self):
        self.replies = [
            codex_stream.StreamResult(
                tool_calls=[
                    _call(
                        "call_1",
                        todos=[
                            {
                                "id": "1",
                                "content": "must not be saved",
                                "status": "pending",
                            }
                        ],
                    )
                ]
            )
        ]

        def fail_checkpoint(*_args, **_kwargs):
            raise ConversationError("disk failed")

        self.agent._store.checkpoint_turn = fail_checkpoint
        with self.assertRaisesRegex(ConversationError, "disk failed"):
            await self.agent.respond("chat", "make a plan")

        self.assertNotIn("todos", self.agent._tool_state["chat"])
        self.assertEqual(len(self.requests), 1)

    async def test_tool_result_write_failure_stops_before_another_model_call(self):
        self.replies = [
            codex_stream.StreamResult(
                tool_calls=[
                    _call(
                        "call_1",
                        todos=[
                            {
                                "id": "1",
                                "content": "already executed",
                                "status": "pending",
                            }
                        ],
                    )
                ]
            )
        ]
        real_checkpoint = self.agent._store.checkpoint_turn

        def checkpoint(*args, **kwargs):
            if kwargs.get("phase") == "tool_completed":
                raise ConversationError("result write failed")
            return real_checkpoint(*args, **kwargs)

        self.agent._store.checkpoint_turn = checkpoint
        with self.assertRaisesRegex(ConversationError, "result write failed"):
            await self.agent.respond("chat", "make a plan")

        self.assertEqual(
            len(self.agent._tool_state["chat"]["todo"].read()),
            1,
        )
        self.assertEqual(len(self.requests), 1)
        with self.assertRaisesRegex(ConversationError, "previous turn"):
            await self.agent.respond("chat", "continue")

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

    async def test_exact_codex_message_is_replayed_before_its_tool_call(self):
        commentary = _message(
            "I will inspect it.",
            phase="commentary",
            item_id="msg_commentary",
        )
        self.replies = [
            codex_stream.StreamResult(
                message_items=[commentary],
                tool_calls=[_call("call_abc")],
            ),
            codex_stream.StreamResult(text="Read it."),
        ]

        await self.agent.respond("chat", "inspect it")

        messages = [item for item in self._sent() if item.get("type") == "message"]
        self.assertEqual(messages[-1], commentary)
        types = [item.get("type") or item.get("role") for item in self._sent()]
        self.assertEqual(
            types[-3:],
            ["message", "function_call", "function_call_output"],
        )

    async def test_oversized_codex_message_id_is_dropped_on_replay(self):
        self.replies = [
            codex_stream.StreamResult(
                message_items=[
                    _message(
                        "I will inspect it.",
                        phase="commentary",
                        item_id="m" * 65,
                    )
                ],
                tool_calls=[_call("call_abc")],
            ),
            codex_stream.StreamResult(text="Read it."),
        ]

        await self.agent.respond("chat", "inspect it")

        message = [item for item in self._sent() if item.get("type") == "message"][-1]
        self.assertNotIn("id", message)
        self.assertEqual(message["phase"], "commentary")

    async def test_commentary_only_response_is_replayed_until_the_final_answer(self):
        commentary = _message("I will inspect it.", phase="commentary")
        self.replies = [
            codex_stream.StreamResult(
                message_items=[commentary],
                needs_continuation=True,
            ),
            codex_stream.StreamResult(text="Inspection complete."),
        ]

        answer = await self.agent.respond("chat", "inspect it")

        self.assertEqual(answer, "Inspection complete.")
        self.assertEqual(len(self.requests), 2)
        self.assertIn(commentary, self._sent())

    async def test_commentary_continuation_stops_after_three_incomplete_responses(self):
        self.replies = [
            codex_stream.StreamResult(
                message_items=[
                    _message(
                        "Still working.",
                        phase="commentary",
                        item_id=f"msg_{index}",
                    )
                ],
                needs_continuation=True,
            )
            for index in range(3)
        ]

        answer = await self.agent.respond("chat", "inspect it")

        self.assertEqual(answer, CODEX_INCOMPLETE_RESPONSE)
        self.assertEqual(len(self.requests), 3)

    async def test_exact_codex_message_survives_to_the_next_user_turn(self):
        final = _message("Done.", phase="final_answer", item_id="msg_final")
        self.replies = [
            codex_stream.StreamResult(text="Done.", message_items=[final]),
        ]
        await self.agent.respond("chat", "first")

        await self.agent.respond("chat", "follow up")

        self.assertIn(final, self._sent())

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

    async def test_attached_image_has_a_path_handle_for_editing(self):
        source = self.root / "attached.png"
        source.write_bytes(bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
            "ae426082"
        ))
        attachment = media.Attachment(
            path=source,
            mime="image/png",
            media_type="image",
        )
        self.replies = [codex_stream.StreamResult(text="I see it.")]

        await self.agent.respond("chat", "edit this", [attachment])

        user = self._sent()[0]
        text_part = next(
            part for part in user["content"] if part.get("type") == "input_text"
        )
        staged = list((self.agent._context_cwd / "inputs").iterdir())
        self.assertEqual(len(staged), 1)
        self.assertIn(
            f"[Image attached at: {staged[0].resolve()}]",
            text_part["text"],
        )
        self.assertNotIn(str(source.resolve()), text_part["text"])
        self.assertTrue(
            any(part.get("type") == "input_image" for part in user["content"])
        )

    async def test_generated_image_is_delivered_when_model_omits_media_tag(self):
        home = self.root / "profile"
        with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(home)}):
            config = Config.load()
        config.workspace_dir.mkdir(parents=True)
        generated = config.workspace_dir / "generated-images" / "result.png"
        generated.parent.mkdir()
        generated.write_bytes(b"png")
        agent = Agent(
            config,
            ConversationStore(self.root / "image-conversations.db"),
        )

        def fake_image(_args, _context):
            return json.dumps(
                {"success": True, "image": str(generated.resolve())}
            )

        agent._registry._tools["image_generate"] = Tool(
            name="image_generate",
            group="image_gen",
            schema={
                "name": "image_generate",
                "description": "test",
                "parameters": {"type": "object"},
            },
            handler=fake_image,
        )
        replies = [
            codex_stream.StreamResult(
                tool_calls=[_call("image_1", name="image_generate", prompt="cat")]
            ),
            codex_stream.StreamResult(text="Here it is."),
        ]

        async def stream_once(
            _request, *, force_refresh, ttfb_timeout, idle_timeout
        ):
            return replies.pop(0)

        agent._stream_once = stream_once
        answer = await agent.respond("chat", "make an image")

        self.assertIn("Here it is.", answer)
        self.assertEqual(answer.count(f"MEDIA:{generated.resolve()}"), 1)

    async def test_model_supplied_media_tag_is_not_duplicated(self):
        home = self.root / "profile-dedup"
        with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(home)}):
            config = Config.load()
        config.workspace_dir.mkdir(parents=True)
        generated = config.workspace_dir / "result.png"
        generated.write_bytes(b"png")
        agent = Agent(
            config,
            ConversationStore(self.root / "dedup-conversations.db"),
        )

        def fake_image(_args, _context):
            return json.dumps(
                {"success": True, "image": str(generated.resolve())}
            )

        agent._registry._tools["image_generate"] = Tool(
            name="image_generate",
            group="image_gen",
            schema={
                "name": "image_generate",
                "description": "test",
                "parameters": {"type": "object"},
            },
            handler=fake_image,
        )
        replies = [
            codex_stream.StreamResult(
                tool_calls=[_call("image_1", name="image_generate", prompt="cat")]
            ),
            codex_stream.StreamResult(
                text=f"Here it is.\nMEDIA:{generated.resolve()}"
            ),
        ]

        async def stream_once(
            _request, *, force_refresh, ttfb_timeout, idle_timeout
        ):
            return replies.pop(0)

        agent._stream_once = stream_once
        answer = await agent.respond("chat", "make an image")

        self.assertEqual(answer.count(f"MEDIA:{generated.resolve()}"), 1)

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
