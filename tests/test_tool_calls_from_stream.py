"""Reading the model's tool calls off the stream.

The hard invariant here is the call id. It goes into the request that is
replayed on every following call of the turn, so it is part of the prefix
OpenAI caches. A random id would make that prefix different every time and
throw the cache away on every turn — which is paid for in latency and in
money, and shows up nowhere as an error.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from pilotage.codex import stream as codex_stream
from pilotage.codex.call_ids import MAX_ITEM_ID_LENGTH, clamp_call_id, deterministic_call_id


def _item(**fields) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.output_item.done", item={"type": "function_call", **fields}
    )


def _message_added(phase: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.output_item.added",
        item={"type": "message", "phase": phase},
    )


def _message_done(
    text: str,
    *,
    phase: str,
    item_id: str,
    status: str = "completed",
) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.output_item.done",
        item={
            "type": "message",
            "role": "assistant",
            "status": status,
            "id": item_id,
            "phase": phase,
            "content": [{"type": "output_text", "text": text}],
        },
    )


def _delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def _completed() -> SimpleNamespace:
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(status="completed", usage=None, id="resp_1", error=None),
    )


class Stream:
    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self._events:
            yield event


async def _consume(events, *, on_text_delta=None) -> codex_stream.StreamResult:
    return await codex_stream.consume_stream(
        Stream(events),
        on_text_delta=on_text_delta,
        ttfb_timeout=1.0,
        idle_timeout=1.0,
    )


class CallIdTests(unittest.TestCase):
    def test_the_same_call_always_gets_the_same_id(self):
        first = deterministic_call_id("todo", '{"a": 1}', 0)
        second = deterministic_call_id("todo", '{"a": 1}', 0)
        self.assertEqual(first, second)

    def test_different_calls_get_different_ids(self):
        self.assertNotEqual(
            deterministic_call_id("todo", '{"a": 1}', 0),
            deterministic_call_id("todo", '{"a": 2}', 0),
        )

    def test_the_same_call_twice_in_one_step_is_told_apart(self):
        self.assertNotEqual(
            deterministic_call_id("todo", "{}", 0), deterministic_call_id("todo", "{}", 1)
        )

    def test_an_id_the_backend_would_refuse_is_shortened(self):
        long_id = "call_" + "x" * 200
        clamped = clamp_call_id(long_id)
        self.assertLessEqual(len(clamped), MAX_ITEM_ID_LENGTH)
        self.assertEqual(clamped, clamp_call_id(long_id))

    def test_an_id_that_fits_is_left_alone(self):
        self.assertEqual(clamp_call_id("call_abc"), "call_abc")


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_call_is_read_off_the_stream(self):
        result = await _consume(
            [_item(name="todo", arguments='{"merge": true}', call_id="call_1"), _completed()]
        )
        self.assertEqual(
            result.tool_calls, [{"call_id": "call_1", "name": "todo", "arguments": '{"merge": true}'}]
        )

    async def test_arguments_are_passed_through_untouched(self):
        """Re-serialising would change the bytes the cache already holds."""
        written = '{ "b" : 2,  "a":1 }'
        result = await _consume([_item(name="todo", arguments=written, call_id="c"), _completed()])
        self.assertEqual(result.tool_calls[0]["arguments"], written)

    async def test_a_call_with_no_arguments_gets_an_empty_object(self):
        result = await _consume([_item(name="todo", arguments="", call_id="c"), _completed()])
        self.assertEqual(result.tool_calls[0]["arguments"], "{}")

    async def test_a_call_with_no_id_gets_one_that_is_the_same_every_time(self):
        events = [_item(name="todo", arguments="{}", call_id=""), _completed()]
        first = await _consume(events)
        second = await _consume(events)
        self.assertTrue(first.tool_calls[0]["call_id"])
        self.assertEqual(first.tool_calls[0]["call_id"], second.tool_calls[0]["call_id"])

    async def test_a_call_with_no_name_is_dropped(self):
        result = await _consume([_item(name="", arguments="{}", call_id="c"), _completed()])
        self.assertEqual(result.tool_calls, [])

    async def test_calls_keep_the_order_the_model_asked_in(self):
        result = await _consume(
            [
                _item(name="todo", arguments='{"v": 1}', call_id="c1"),
                _item(name="todo", arguments='{"v": 2}', call_id="c2"),
                _completed(),
            ]
        )
        self.assertEqual([call["call_id"] for call in result.tool_calls], ["c1", "c2"])

    async def test_an_answer_with_no_calls_has_none(self):
        result = await _consume(
            [SimpleNamespace(type="response.output_text.delta", delta="Hello."), _completed()]
        )
        self.assertEqual(result.tool_calls, [])

    async def test_commentary_and_final_answer_are_kept_separate(self):
        visible = []
        result = await _consume(
            [
                _message_added("commentary"),
                _delta("I will inspect the repository."),
                _message_done(
                    "I will inspect the repository.",
                    phase="commentary",
                    item_id="msg_commentary",
                ),
                _message_added("final_answer"),
                _delta("Everything is ready."),
                _message_done(
                    "Everything is ready.",
                    phase="final_answer",
                    item_id="msg_final",
                ),
                _completed(),
            ],
            on_text_delta=visible.append,
        )

        self.assertEqual(result.text, "Everything is ready.")
        self.assertEqual(visible, ["Everything is ready."])
        self.assertFalse(result.needs_continuation)
        self.assertEqual(
            result.message_items,
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "id": "msg_commentary",
                    "phase": "commentary",
                    "content": [
                        {"type": "output_text", "text": "I will inspect the repository."}
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "id": "msg_final",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "Everything is ready."}],
                },
            ],
        )

    async def test_commentary_only_is_an_incomplete_turn_not_an_answer(self):
        result = await _consume(
            [
                _message_added("commentary"),
                _delta("I will call the tool now."),
                _message_done(
                    "I will call the tool now.",
                    phase="commentary",
                    item_id="msg_commentary",
                ),
                _completed(),
            ]
        )

        self.assertEqual(result.text, "")
        self.assertTrue(result.needs_continuation)
        self.assertEqual(result.message_items[0]["phase"], "commentary")

    async def test_commentary_before_a_tool_is_replayed_but_does_not_delay_the_tool(self):
        result = await _consume(
            [
                _message_added("commentary"),
                _delta("I will inspect it."),
                _message_done(
                    "I will inspect it.",
                    phase="commentary",
                    item_id="msg_commentary",
                ),
                _item(name="todo", arguments="{}", call_id="call_1"),
                _completed(),
            ]
        )

        self.assertEqual(result.text, "")
        self.assertFalse(result.needs_continuation)
        self.assertEqual(result.message_items[0]["phase"], "commentary")
        self.assertEqual(result.tool_calls[0]["call_id"], "call_1")

    async def test_analysis_phase_is_also_hidden_from_the_answer(self):
        result = await _consume(
            [
                _message_added("analysis"),
                _delta("Private analysis."),
                _message_done(
                    "Private analysis.",
                    phase="analysis",
                    item_id="msg_analysis",
                    status="in-progress",
                ),
                _completed(),
            ]
        )

        self.assertEqual(result.text, "")
        self.assertTrue(result.needs_continuation)
        self.assertEqual(result.message_items[0]["status"], "in_progress")


class CacheKeyTests(unittest.TestCase):
    def test_the_key_does_not_depend_on_the_order_tools_were_registered(self):
        a = {"type": "function", "name": "alpha"}
        b = {"type": "function", "name": "beta"}
        self.assertEqual(
            codex_stream.content_cache_key("do things", [a, b], "chat"),
            codex_stream.content_cache_key("do things", [b, a], "chat"),
        )

    def test_the_key_changes_when_the_tools_change(self):
        a = {"type": "function", "name": "alpha"}
        self.assertNotEqual(
            codex_stream.content_cache_key("do things", [a], "chat"),
            codex_stream.content_cache_key("do things", None, "chat"),
        )

    def test_two_chats_do_not_share_a_cache_key(self):
        self.assertNotEqual(
            codex_stream.content_cache_key("do things", None, "chat_a"),
            codex_stream.content_cache_key("do things", None, "chat_b"),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
