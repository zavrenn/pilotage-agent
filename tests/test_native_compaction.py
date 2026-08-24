"""Native Responses compaction on Pilotage's fixed ChatGPT Codex route."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from pilotage.codex import compaction
from pilotage.codex import stream as codex_stream


class ModelGateTests(unittest.TestCase):
    def test_only_the_verified_gpt_56_family_is_eligible(self):
        for model in ("gpt-5.6", "gpt-5.6-sol", "GPT-5.6-2026-07-15"):
            with self.subTest(model=model):
                self.assertTrue(compaction.is_native_compaction_model(model))
        for model in ("gpt-5.1", "gpt-5.2", "gpt-5.3-codex", "gpt-4o", ""):
            with self.subTest(model=model):
                self.assertFalse(compaction.is_native_compaction_model(model))

    def test_eligible_request_gets_the_configured_directive(self):
        request = codex_stream.build_request(
            model="gpt-5.6-sol",
            instructions="test",
            input_items=[{"role": "user", "content": "hello"}],
            session_id="chat",
            reasoning_effort="medium",
            native_compaction_enabled=True,
            compact_threshold=123_456,
        )
        self.assertEqual(
            request["context_management"],
            [{"type": "compaction", "compact_threshold": 123_456}],
        )

    def test_disabled_or_ineligible_request_omits_the_directive(self):
        for model, enabled in (("gpt-5.6-sol", False), ("gpt-5.2", True)):
            with self.subTest(model=model, enabled=enabled):
                request = codex_stream.build_request(
                    model=model,
                    instructions="test",
                    input_items=[{"role": "user", "content": "hello"}],
                    session_id="chat",
                    reasoning_effort="medium",
                    native_compaction_enabled=enabled,
                )
                self.assertNotIn("context_management", request)


class CheckpointReplayTests(unittest.TestCase):
    def _items(self):
        return [
            {"role": "user", "content": "goal: build the runner"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "constraint: no polling"},
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "out"},
            {"type": "compaction", "encrypted_content": "opaque"},
            {"role": "assistant", "content": "checkpointed"},
            {"role": "user", "content": "next"},
        ]

    def test_checkpoint_leads_and_pre_checkpoint_tool_traffic_is_dropped(self):
        output = compaction.prune_pre_checkpoint_items(self._items())
        self.assertEqual(
            output[0], {"type": "compaction", "encrypted_content": "opaque"}
        )
        self.assertNotIn({"role": "assistant", "content": "ack"}, output)
        self.assertFalse(any(item.get("call_id") == "c1" for item in output))
        self.assertEqual(
            [item["content"] for item in output if item.get("role") == "user"],
            ["goal: build the runner", "constraint: no polling", "next"],
        )

    def test_retention_budget_keeps_newest_users_first(self):
        old = {"role": "user", "content": "x" * 4000}
        newer = {"role": "user", "content": "y" * 2000}
        output = compaction.prune_pre_checkpoint_items(
            [old, newer, {"type": "compaction", "encrypted_content": "cp"}],
            retained_user_token_budget=600,
        )
        users = [item["content"] for item in output if item.get("role") == "user"]
        self.assertEqual(users, ["x" * 400, "y" * 2000])

    def test_image_only_user_turn_survives_checkpoint_pruning(self):
        image_only = {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
            ],
        }
        checkpoint = {"type": "compaction", "encrypted_content": "opaque"}

        output = compaction.prune_pre_checkpoint_items([image_only, checkpoint])

        self.assertEqual(output, [checkpoint, image_only])

    def test_current_eligibility_gates_checkpoint_replay_and_pruning(self):
        active = codex_stream.build_request(
            model="gpt-5.6-sol",
            instructions="test",
            input_items=self._items(),
            session_id="chat",
            reasoning_effort="medium",
            native_compaction_enabled=True,
        )
        self.assertEqual(active["input"][0]["type"], "compaction")

        inactive = codex_stream.build_request(
            model="gpt-5.2",
            instructions="test",
            input_items=self._items(),
            session_id="chat",
            reasoning_effort="medium",
            native_compaction_enabled=True,
        )
        self.assertFalse(any(item.get("type") == "compaction" for item in inactive["input"]))
        self.assertIn({"role": "assistant", "content": "ack"}, inactive["input"])

    def test_only_valid_opaque_checkpoints_are_persisted(self):
        self.assertEqual(
            compaction.persistent_compaction_items(
                [
                    {"type": "reasoning", "encrypted_content": "thought"},
                    {"type": "compaction", "encrypted_content": "opaque"},
                    {"type": "compaction", "encrypted_content": ""},
                ]
            ),
            [{"type": "compaction", "encrypted_content": "opaque"}],
        )


class _Stream:
    def __init__(self, events):
        self.events = events

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self.events:
            yield event


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_compaction_output_is_captured_without_extra_fields(self):
        events = [
            SimpleNamespace(
                type="response.output_item.done",
                item={"type": "compaction", "encrypted_content": "opaque", "id": "cp_1"},
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", usage=None, id="r1", error=None),
            ),
        ]
        result = await codex_stream.consume_stream(
            _Stream(events), ttfb_timeout=1.0, idle_timeout=1.0
        )
        self.assertEqual(
            result.reasoning_items,
            [{"type": "compaction", "encrypted_content": "opaque"}],
        )


if __name__ == "__main__":
    unittest.main()
