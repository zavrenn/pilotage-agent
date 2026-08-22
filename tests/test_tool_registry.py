"""What happens when the model asks for a tool.

Two properties carry the whole loop. A handler must never raise into the turn,
because an exception there costs the person waiting their answer rather than
costing the model one failed call. And a result must be bounded, because it is
replayed on every later request of the conversation — one unbounded result is
paid for again on every turn until the chat ends.
"""

from __future__ import annotations

import asyncio
import json
import unittest

from pilotage.settings import ConfigError, Settings
from pilotage.tools import Registry, Tool, ToolContext, build_registry, enabled_groups, run_calls
from pilotage.tools.registry import DEFAULT_STEP_BUDGET_CHARS, tool_result


def _echo(args, context):
    return tool_result(args)


def _context() -> ToolContext:
    return ToolContext(chat_id="chat", config=None)


def _tool(name: str, handler=_echo, group: str = "test", **kwargs) -> Tool:
    return Tool(
        name=name,
        group=group,
        schema={"name": name, "description": f"The {name} tool.", "parameters": {"type": "object"}},
        handler=handler,
        **kwargs,
    )


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = Registry()
        self.registry.register(_tool("echo"))

    def test_a_tool_cannot_be_registered_twice(self):
        with self.assertRaises(ValueError):
            self.registry.register(_tool("echo"))

    def test_only_enabled_groups_are_offered_to_the_model(self):
        self.registry.register(_tool("shell", group="terminal"))
        offered = [item["name"] for item in self.registry.definitions(["test"])]
        self.assertEqual(offered, ["echo"])

    def test_nothing_enabled_means_no_tools_at_all(self):
        self.assertEqual(self.registry.definitions([]), [])

    def test_the_list_does_not_depend_on_import_order(self):
        """The tool list is part of the cached request prefix."""
        first = Registry()
        for name in ("zulu", "alpha", "mike"):
            first.register(_tool(name))
        second = Registry()
        for name in ("mike", "zulu", "alpha"):
            second.register(_tool(name))
        self.assertEqual(first.definitions(["test"]), second.definitions(["test"]))

    def test_a_definition_is_the_flat_shape_the_responses_api_takes(self):
        definition = self.registry.definitions(["test"])[0]
        self.assertEqual(definition["type"], "function")
        self.assertEqual(definition["name"], "echo")
        self.assertEqual(definition["description"], "The echo tool.")
        self.assertIn("parameters", definition)
        self.assertNotIn("function", definition)


class DispatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = Registry()
        self.registry.register(_tool("echo"))

    async def _run(self, name: str, arguments: str, **kwargs) -> dict:
        result = await self.registry.dispatch(name, arguments, _context(), **kwargs)
        return json.loads(result)

    async def test_a_call_returns_what_the_handler_returned(self):
        self.assertEqual(await self._run("echo", '{"a": 1}'), {"a": 1})

    async def test_no_arguments_at_all_is_an_empty_object(self):
        self.assertEqual(await self._run("echo", ""), {})

    async def test_a_tool_the_model_invented_comes_back_as_an_error(self):
        self.assertIn("error", await self._run("nonexistent", "{}"))

    async def test_a_registered_but_disabled_tool_cannot_execute(self):
        called = False

        def _dangerous(args, context):
            nonlocal called
            called = True
            return tool_result(ok=True)

        self.registry.register(_tool("dangerous", handler=_dangerous, group="terminal"))
        answer = await self._run("dangerous", "{}", allowed_groups=["test"])
        self.assertIn("disabled", answer["error"])
        self.assertFalse(called)

    async def test_arguments_that_are_not_json_come_back_as_an_error(self):
        self.assertIn("error", await self._run("echo", "{not json"))

    async def test_arguments_that_are_not_an_object_come_back_as_an_error(self):
        self.assertIn("error", await self._run("echo", "[1, 2, 3]"))

    async def test_a_handler_that_raises_does_not_end_the_turn(self):
        def _explode(args, context):
            raise RuntimeError("the disk is on fire")

        self.registry.register(_tool("boom", handler=_explode))
        with self.assertLogs("pilotage.tools.registry", level="ERROR"):
            answer = await self._run("boom", "{}")
        self.assertIn("the disk is on fire", answer["error"])

    async def test_a_handler_that_returns_the_wrong_thing_is_an_error_not_a_crash(self):
        self.registry.register(_tool("wrong", handler=lambda args, context: {"not": "text"}))
        with self.assertLogs("pilotage.tools.registry", level="ERROR"):
            self.assertIn("error", await self._run("wrong", "{}"))

    async def test_an_async_handler_works_the_same(self):
        async def _slow(args, context):
            await asyncio.sleep(0)
            return tool_result(ok=True)

        self.registry.register(_tool("slow", handler=_slow))
        self.assertEqual(await self._run("slow", "{}"), {"ok": True})

    async def test_a_huge_error_message_is_bounded(self):
        def _verbose(args, context):
            raise RuntimeError("x" * 100_000)

        self.registry.register(_tool("verbose", handler=_verbose))
        with self.assertLogs("pilotage.tools.registry", level="ERROR"):
            answer = await self._run("verbose", "{}")
        self.assertLess(len(answer["error"]), 5_000)

    async def test_a_huge_result_is_cut_before_it_reaches_the_conversation(self):
        self.registry.register(_tool("flood", handler=lambda args, context: "y" * 50_000))
        result = await self.registry.dispatch("flood", "{}", _context(), max_result_chars=1_000)
        self.assertLess(len(result), 1_200)
        self.assertIn("truncated", result)

    async def test_the_head_of_a_cut_result_is_what_survives(self):
        """The answer is usually at the start of a file or a command's output."""
        self.registry.register(
            _tool("flood", handler=lambda args, context: "START" + "y" * 50_000)
        )
        result = await self.registry.dispatch("flood", "{}", _context(), max_result_chars=1_000)
        self.assertTrue(result.startswith("START"))

    async def test_a_tool_carries_its_own_limit(self):
        self.registry.register(
            _tool("small", handler=lambda args, context: "z" * 5_000, max_result_chars=100)
        )
        result = await self.registry.dispatch("small", "{}", _context())
        self.assertLess(len(result), 300)

    async def test_a_cut_json_result_remains_valid_json(self):
        self.registry.register(
            _tool("json_flood", handler=lambda args, context: tool_result(value="\\" * 5_000))
        )
        result = await self.registry.dispatch(
            "json_flood", "{}", _context(), max_result_chars=1_000
        )
        parsed = json.loads(result)
        self.assertTrue(parsed["truncated"])
        self.assertIn("prefix", parsed)


class StepTests(unittest.IsolatedAsyncioTestCase):
    """Several calls in one step: the model asks for them together on purpose."""

    def setUp(self):
        self.registry = Registry()
        self.registry.register(_tool("echo"))

    def _call(self, value):
        return {"call_id": f"call_{value}", "name": "echo", "arguments": json.dumps({"v": value})}

    async def test_no_calls_is_no_work(self):
        self.assertEqual(await run_calls(self.registry, [], _context()), [])

    async def test_execution_enforces_the_same_groups_as_the_tool_offer(self):
        results = await run_calls(
            self.registry,
            [self._call(1)],
            _context(),
            allowed_groups=["terminal"],
        )
        self.assertIn("disabled", json.loads(results[0])["error"])

    async def test_results_come_back_in_the_order_they_were_asked_for(self):
        """They run together, but the request built from them must be stable."""
        both_started = asyncio.Event()
        started = []
        finished = []

        async def _uneven(args, context):
            started.append(args["v"])
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)
            if args["v"] == 0:
                await asyncio.sleep(0.01)
            finished.append(args["v"])
            return tool_result(v=args["v"])

        self.registry = Registry()
        self.registry.register(_tool("echo", handler=_uneven))
        results = await run_calls(self.registry, [self._call(0), self._call(1)], _context())
        self.assertEqual([json.loads(r)["v"] for r in results], [0, 1])
        self.assertEqual(finished, [1, 0])

    async def test_one_step_cannot_spend_more_than_its_budget(self):
        self.registry = Registry()
        self.registry.register(_tool("echo", handler=lambda args, context: "y" * 900))
        calls = [self._call(n) for n in range(5)]
        results = await run_calls(self.registry, calls, _context(), step_budget_chars=1_000)
        self.assertEqual(len(results), 5)
        self.assertLess(sum(len(r) for r in results), 1_500)

    async def test_a_failing_call_does_not_stop_the_others(self):
        def _sometimes(args, context):
            if args["v"] == 0:
                raise RuntimeError("no")
            return tool_result(v=args["v"])

        self.registry = Registry()
        self.registry.register(_tool("echo", handler=_sometimes))
        with self.assertLogs("pilotage.tools.registry", level="ERROR"):
            results = await run_calls(self.registry, [self._call(0), self._call(1)], _context())
        self.assertIn("error", json.loads(results[0]))
        self.assertEqual(json.loads(results[1])["v"], 1)

    async def test_the_default_budget_is_the_one_we_chose(self):
        self.assertEqual(DEFAULT_STEP_BUDGET_CHARS, 200_000)


class EnabledGroupTests(unittest.TestCase):
    """Which tools an agent may use is the operator's decision, not the build's."""

    def setUp(self):
        self.registry = Registry()
        self.registry.register(_tool("echo", group="todo"))
        self.registry.register(_tool("shell", group="terminal"))
        self.registry.register(_tool("fetch", group="web"))

    def _groups(self, data: dict, channel: str = "") -> list:
        settings = Settings(data)
        if channel:
            settings = settings.for_channel(channel)
        return enabled_groups(settings, self.registry)

    def test_an_empty_file_leaves_every_group_on(self):
        self.assertEqual(self._groups({}), ["terminal", "todo", "web"])

    def test_enabled_names_exactly_what_is_on(self):
        self.assertEqual(self._groups({"tools": {"enabled": ["todo"]}}), ["todo"])

    def test_disabled_takes_a_group_away(self):
        self.assertEqual(self._groups({"tools": {"disabled": ["terminal"]}}), ["todo", "web"])

    def test_disabled_wins_over_enabled(self):
        """Switching something off is the safety-side edit; it must not be lost."""
        data = {"tools": {"enabled": ["todo", "terminal"], "disabled": ["terminal"]}}
        self.assertEqual(self._groups(data), ["todo"])

    def test_an_unknown_enabled_group_stops_startup(self):
        with self.assertRaisesRegex(ConfigError, "tools.enabled.*typo"):
            self._groups({"tools": {"enabled": ["todo", "typo"]}})

    def test_an_unknown_disabled_group_stops_startup(self):
        with self.assertRaisesRegex(ConfigError, "tools.disabled.*typo"):
            self._groups({"tools": {"disabled": ["typo"]}})

    def test_a_channel_can_run_with_fewer_tools(self):
        data = {"channels": {"whatsapp": {"tools": {"disabled": ["terminal"]}}}}
        self.assertEqual(self._groups(data, "whatsapp"), ["todo", "web"])
        self.assertEqual(self._groups(data), ["terminal", "todo", "web"])


class BuiltRegistryTests(unittest.TestCase):
    def test_the_registry_this_build_ships_has_the_tools_built_so_far(self):
        registry = build_registry()
        self.assertIn("file", registry.groups())
        self.assertIn("skills", registry.groups())
        self.assertIn("memory", registry.groups())
        self.assertIn("todo", registry.groups())
        self.assertIn("terminal", registry.groups())
        self.assertEqual(
            {"patch", "read_file", "search_files", "write_file"},
            set(registry.names(["file"])),
        )
        self.assertIsNotNone(registry.get("todo"))
        self.assertIsNotNone(registry.get("memory"))
        self.assertIsNotNone(registry.get("terminal"))
        self.assertEqual(
            {"skill_view", "skills_list"},
            set(registry.names(["skills"])),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
