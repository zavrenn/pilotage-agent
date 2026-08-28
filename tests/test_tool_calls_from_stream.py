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
from unittest import mock

from pilotage.codex import stream as codex_stream
from pilotage.codex.call_ids import MAX_ITEM_ID_LENGTH, clamp_call_id, deterministic_call_id


def _item(**fields) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.output_item.done", item={"type": "function_call", **fields}
    )


def _function_added(
    *,
    item_id: str,
    call_id: str,
    name: str,
    arguments: str = "",
    output_index=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.output_item.added",
        output_index=output_index,
        item={
            "type": "function_call",
            "id": item_id,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        },
    )


def _arguments_delta(item_id: str, delta: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.function_call_arguments.delta",
        item_id=item_id,
        delta=delta,
    )


def _arguments_done(item_id: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.function_call_arguments.done",
        item_id=item_id,
        arguments=arguments,
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


class ReplayedFunctionNameTests(unittest.TestCase):
    def _request(self, name, *, tool_name="live_tool"):
        return codex_stream.build_request(
            model="gpt-5.6-sol",
            instructions="test",
            input_items=[
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": name,
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                },
            ],
            session_id="chat",
            reasoning_effort="medium",
            tools=[
                {
                    "type": "function",
                    "name": tool_name,
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )

    def test_valid_replayed_names_pass_through_unchanged(self):
        for name in ("web_search", "exec-command", "x" * 64):
            self.assertEqual(self._request(name)["input"][0]["name"], name)

    def test_invalid_replayed_names_are_coerced_without_breaking_pairing(self):
        request = self._request("run weird.tool!")

        self.assertEqual(request["input"][0]["name"], "run_weird_tool")
        self.assertEqual(request["input"][0]["call_id"], "call_1")
        self.assertEqual(request["input"][1]["call_id"], "call_1")

    def test_degenerate_replayed_names_never_become_empty(self):
        self.assertEqual(self._request("日本語")["input"][0]["name"], "fn")
        self.assertLessEqual(
            len(self._request("bad." * 100)["input"][0]["name"]),
            64,
        )

    def test_live_tool_definition_names_are_not_rewritten(self):
        request = self._request("bad.name", tool_name="schema.name")

        self.assertEqual(request["tools"][0]["name"], "schema.name")


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

    async def test_a_pending_call_settles_when_item_done_is_omitted(self):
        result = await _consume(
            [
                _function_added(item_id="fc_1", call_id="c1", name="todo"),
                _arguments_delta("fc_1", '{"merge":'),
                _arguments_delta("fc_1", "true}"),
                _completed(),
            ]
        )

        self.assertEqual(
            result.tool_calls,
            [{"call_id": "c1", "name": "todo", "arguments": '{"merge":true}'}],
        )

    async def test_arguments_done_is_authoritative_for_a_pending_call(self):
        result = await _consume(
            [
                _function_added(item_id="fc_1", call_id="c1", name="todo"),
                _arguments_delta("fc_1", '{"partial":'),
                _arguments_done("fc_1", '{"final": true}'),
                _completed(),
            ]
        )

        self.assertEqual(result.tool_calls[0]["arguments"], '{"final": true}')

    async def test_zero_argument_pending_call_settles_with_an_empty_object(self):
        result = await _consume(
            [_function_added(item_id="fc_1", call_id="c1", name="todo"), _completed()]
        )

        self.assertEqual(result.tool_calls[0]["arguments"], "{}")

    async def test_pending_call_is_not_settled_without_successful_completion(self):
        incomplete = SimpleNamespace(
            type="response.incomplete",
            response=SimpleNamespace(
                status="incomplete", usage=None, id="resp_1", error=None
            ),
        )
        result = await _consume(
            [_function_added(item_id="fc_1", call_id="c1", name="todo"), incomplete]
        )

        self.assertEqual(result.tool_calls, [])

    async def test_item_done_remains_authoritative_and_is_not_duplicated(self):
        result = await _consume(
            [
                _function_added(item_id="fc_1", call_id="c1", name="todo"),
                _arguments_delta("fc_1", '{"partial":'),
                _item(
                    id="fc_1",
                    name="todo",
                    arguments='{"done": true}',
                    call_id="c1",
                ),
                _completed(),
            ]
        )

        self.assertEqual(
            result.tool_calls,
            [{"call_id": "c1", "name": "todo", "arguments": '{"done": true}'}],
        )

    async def test_pending_and_done_calls_keep_output_index_order(self):
        done_second = SimpleNamespace(
            type="response.output_item.done",
            output_index=1,
            item={
                "type": "function_call",
                "id": "fc_2",
                "call_id": "c2",
                "name": "second",
                "arguments": '{"step": 2}',
            },
        )
        result = await _consume(
            [
                _function_added(
                    item_id="fc_1", call_id="c1", name="first", output_index=0
                ),
                _arguments_delta("fc_1", '{"step": 1}'),
                _function_added(
                    item_id="fc_2", call_id="c2", name="second", output_index=1
                ),
                done_second,
                _completed(),
            ]
        )

        self.assertEqual([call["call_id"] for call in result.tool_calls], ["c1", "c2"])

    async def test_missing_indexes_use_first_observed_order(self):
        result = await _consume(
            [
                _function_added(item_id="fc_1", call_id="c1", name="first"),
                _function_added(item_id="fc_2", call_id="c2", name="second"),
                _item(id="fc_1", name="first", arguments="{}", call_id="c1"),
                _completed(),
            ]
        )

        self.assertEqual([call["call_id"] for call in result.tool_calls], ["c1", "c2"])

    async def test_eof_does_not_turn_a_pending_call_into_authority(self):
        with self.assertRaises(codex_stream.CodexStreamError):
            await _consume(
                [_function_added(item_id="fc_1", call_id="c1", name="todo")]
            )

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


class SdkTransformBypassTests(unittest.TestCase):
    def _request(self):
        return {
            "model": "gpt-test",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "tools": [{"type": "function", "name": "terminal", "parameters": {}}],
            "store": False,
        }

    def test_plain_json_payloads_are_moved_without_mutating_the_request(self):
        request = self._request()
        original_input = request["input"]

        bypassed = codex_stream._bypass_sdk_request_transform(request)

        self.assertNotIn("input", bypassed)
        self.assertNotIn("tools", bypassed)
        self.assertIs(bypassed["extra_body"]["input"], original_input)
        self.assertEqual(bypassed["extra_body"]["tools"], request["tools"])
        self.assertIs(request["input"], original_input)
        self.assertNotIn("extra_body", request)

    def test_existing_extra_body_keeps_precedence(self):
        request = self._request()
        caller_extra = {"input": "explicit", "prompt_cache_retention": "24h"}
        request["extra_body"] = caller_extra

        bypassed = codex_stream._bypass_sdk_request_transform(request)

        self.assertEqual(bypassed["extra_body"]["input"], "explicit")
        self.assertEqual(bypassed["extra_body"]["tools"], request["tools"])
        self.assertEqual(caller_extra, {"input": "explicit", "prompt_cache_retention": "24h"})

    def test_non_json_bulk_field_stays_on_the_typed_path(self):
        request = self._request()
        request["input"] = [{"content": object()}]

        bypassed = codex_stream._bypass_sdk_request_transform(request)

        self.assertIs(bypassed["input"], request["input"])
        self.assertEqual(bypassed["extra_body"], {"tools": request["tools"]})

    def test_scalar_input_stays_on_the_typed_path(self):
        request = {"model": "gpt-test", "input": "hello"}

        self.assertIs(codex_stream._bypass_sdk_request_transform(request), request)

    def test_escape_hatch_restores_the_sdk_transform(self):
        request = self._request()
        with mock.patch.dict(
            "os.environ", {"PILOTAGE_CODEX_SDK_TRANSFORM": "1"}, clear=False
        ):
            self.assertIs(codex_stream._bypass_sdk_request_transform(request), request)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
