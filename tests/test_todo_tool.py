"""The task list the model keeps while it works.

Copied from Hermes, so these tests are about the contract rather than the
implementation: a plan the model states is the plan, a malformed item becomes
something visible instead of an error, and the list belongs to one chat.
"""

from __future__ import annotations

import json
import unittest

from pilotage.tools import ToolContext
from pilotage.tools.todo import MAX_TODO_CONTENT_CHARS, MAX_TODO_ITEMS, TodoStore, handle


def _item(item_id: str, content: str = "do it", status: str = "pending") -> dict:
    return {"id": item_id, "content": content, "status": status}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store = TodoStore()

    def test_a_new_list_is_empty(self):
        self.assertEqual(self.store.read(), [])
        self.assertFalse(self.store.has_items())

    def test_writing_replaces_the_plan_by_default(self):
        self.store.write([_item("1"), _item("2")])
        self.store.write([_item("3")])
        self.assertEqual([item["id"] for item in self.store.read()], ["3"])

    def test_merging_updates_one_step_and_keeps_the_rest(self):
        self.store.write([_item("1"), _item("2")])
        self.store.write([{"id": "1", "status": "completed"}], merge=True)
        items = {item["id"]: item for item in self.store.read()}
        self.assertEqual(items["1"]["status"], "completed")
        self.assertEqual(items["1"]["content"], "do it")
        self.assertEqual(items["2"]["status"], "pending")

    def test_merging_adds_what_is_new(self):
        self.store.write([_item("1")])
        self.store.write([_item("2")], merge=True)
        self.assertEqual([item["id"] for item in self.store.read()], ["1", "2"])

    def test_the_step_being_worked_on_comes_first(self):
        self.store.write([_item("1"), _item("2", status="in_progress")])
        self.assertEqual([item["id"] for item in self.store.read()], ["2", "1"])

    def test_a_repeated_id_is_written_once(self):
        self.store.write([_item("1", "first"), _item("1", "second")])
        items = self.store.read()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["content"], "second")

    def test_an_unknown_status_becomes_pending_rather_than_an_error(self):
        self.store.write([_item("1", status="almost")])
        self.assertEqual(self.store.read()[0]["status"], "pending")

    def test_an_item_that_is_not_an_object_becomes_a_visible_placeholder(self):
        self.store.write(["not an item"])
        self.assertEqual(self.store.read()[0]["content"], "(invalid item)")

    def test_an_item_with_no_description_says_so(self):
        self.store.write([{"id": "1"}])
        self.assertEqual(self.store.read()[0]["content"], "(no description)")

    def test_one_oversized_item_cannot_take_over_the_conversation(self):
        self.store.write([_item("1", "x" * 20_000)])
        self.assertLessEqual(len(self.store.read()[0]["content"]), MAX_TODO_CONTENT_CHARS)

    def test_the_list_itself_is_bounded(self):
        self.store.write([_item(str(n)) for n in range(MAX_TODO_ITEMS + 50)])
        self.assertEqual(len(self.store.read()), MAX_TODO_ITEMS)

    def test_the_caller_cannot_edit_the_list_behind_its_back(self):
        self.store.write([_item("1")])
        self.store.read()[0]["status"] = "completed"
        self.assertEqual(self.store.read()[0]["status"], "pending")


class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.context = ToolContext(chat_id="chat", config=None)

    def _call(self, **args) -> dict:
        return json.loads(handle(args, self.context))

    def test_a_call_with_nothing_reads_the_list(self):
        self._call(todos=[_item("1")])
        self.assertEqual(len(self._call()["todos"]), 1)

    def test_a_write_answers_with_the_whole_list(self):
        answer = self._call(todos=[_item("1"), _item("2", status="completed")])
        self.assertEqual(len(answer["todos"]), 2)
        self.assertEqual(answer["summary"]["total"], 2)
        self.assertEqual(answer["summary"]["completed"], 1)
        self.assertEqual(answer["summary"]["pending"], 1)

    def test_a_list_sent_as_a_json_string_still_works(self):
        answer = self._call(todos=json.dumps([_item("1")]))
        self.assertEqual(len(answer["todos"]), 1)

    def test_a_string_that_is_not_json_is_an_error_the_model_can_read(self):
        self.assertIn("error", self._call(todos="just do the thing"))

    def test_something_that_is_not_a_list_is_an_error(self):
        self.assertIn("error", self._call(todos={"id": "1"}))

    def test_the_list_lives_on_the_chat_not_on_the_tool(self):
        self._call(todos=[_item("1")])
        other = ToolContext(chat_id="other", config=None)
        self.assertEqual(json.loads(handle({}, other))["todos"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
