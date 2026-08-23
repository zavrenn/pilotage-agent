"""Contract for the Hermes-derived DDGS web-search tool."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

from pilotage.tools import ToolContext, build_registry
from pilotage.tools import web


def _fake_ddgs(results=()):
    module = types.ModuleType("ddgs")

    class FakeDDGS:
        timeout = None
        query = ""
        max_results = 0

        def __init__(self, *, timeout):
            type(self).timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def text(self, query, *, max_results):
            type(self).query = query
            type(self).max_results = max_results
            return iter(results)

    module.DDGS = FakeDDGS
    return module, FakeDDGS


class ResultNormalizationTests(unittest.TestCase):
    def test_ddgs_hits_are_normalized_to_hermes_shape(self):
        module, fake = _fake_ddgs(
            [
                {"title": "A", "href": "https://a.example", "body": "First"},
                {"title": "B", "url": "https://b.example", "body": "Second"},
            ]
        )
        with mock.patch.dict(sys.modules, {"ddgs": module}):
            results = web._run_ddgs_search("pilotage", 2)

        self.assertEqual(fake.timeout, 10)
        self.assertEqual(fake.query, "pilotage")
        self.assertEqual(fake.max_results, 2)
        self.assertEqual(
            results,
            [
                {
                    "title": "A",
                    "url": "https://a.example",
                    "description": "First",
                    "position": 1,
                },
                {
                    "title": "B",
                    "url": "https://b.example",
                    "description": "Second",
                    "position": 2,
                },
            ],
        )

    def test_result_limit_is_enforced_even_if_ddgs_overproduces(self):
        module, _ = _fake_ddgs(
            [
                {"title": str(index), "href": f"https://{index}.example", "body": ""}
                for index in range(5)
            ]
        )
        with mock.patch.dict(sys.modules, {"ddgs": module}):
            results = web._run_ddgs_search("q", 2)
        self.assertEqual(len(results), 2)


class SearchContractTests(unittest.TestCase):
    def test_missing_dependency_is_a_visible_tool_failure(self):
        with mock.patch.dict(sys.modules, {"ddgs": None}):
            result = web._search("q")
        self.assertFalse(result["success"])
        self.assertIn("scripts/install.sh", result["error"])

    def test_limit_is_clamped_to_hermes_maximum(self):
        module, _ = _fake_ddgs()
        with (
            mock.patch.dict(sys.modules, {"ddgs": module}),
            mock.patch.object(web, "_run_ddgs_search_bounded", return_value=[]) as search,
        ):
            result = web._search("q", 50_000)
        self.assertTrue(result["success"])
        search.assert_called_once_with("q", 100)

    def test_provider_failure_is_returned_as_data(self):
        module, _ = _fake_ddgs()
        with (
            mock.patch.dict(sys.modules, {"ddgs": module}),
            mock.patch.object(
                web,
                "_run_ddgs_search_bounded",
                side_effect=RuntimeError("rate limited"),
            ),
        ):
            result = web._search("q")
        self.assertFalse(result["success"])
        self.assertIn("rate limited", result["error"])

    def test_timeout_has_a_specific_recovery_message(self):
        module, _ = _fake_ddgs()
        with (
            mock.patch.dict(sys.modules, {"ddgs": module}),
            mock.patch.object(
                web,
                "_run_ddgs_search_bounded",
                side_effect=TimeoutError("late"),
            ),
        ):
            result = web._search("q")
        self.assertFalse(result["success"])
        self.assertIn("timed out", result["error"])


class ProcessIsolationTests(unittest.TestCase):
    def setUp(self):
        self.original_hook = web._test_hook
        self.original_timeout = web._SEARCH_TIMEOUT_SECONDS
        self.original_reap = web._WORKER_REAP_SECONDS
        self.addCleanup(self._restore)

    def _restore(self):
        web._test_hook = self.original_hook
        web._SEARCH_TIMEOUT_SECONDS = self.original_timeout
        web._WORKER_REAP_SECONDS = self.original_reap

    def test_fast_worker_result_crosses_the_process_boundary(self):
        web._test_hook = "success"
        results = web._run_ddgs_search_bounded("q", 5)
        self.assertEqual(results[0]["url"], "https://example.com")
        self.assertIsNotNone(web._last_worker_proc)
        self.assertIsNotNone(web._last_worker_proc.poll())

    def test_hung_worker_is_killed_and_reaped_at_the_deadline(self):
        web._test_hook = "sleep"
        web._SEARCH_TIMEOUT_SECONDS = 0.3
        web._WORKER_REAP_SECONDS = 0.5

        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            web._run_ddgs_search_bounded("q", 5)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 5.0)
        self.assertIsNotNone(web._last_worker_proc)
        self.assertIsNotNone(web._last_worker_proc.poll())

    def test_worker_failure_envelope_is_not_mistaken_for_empty_results(self):
        web._test_hook = "error"
        with self.assertRaisesRegex(RuntimeError, "boom"):
            web._run_ddgs_search_bounded("q", 5)

    def test_worker_environment_does_not_inherit_agent_secrets(self):
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "secret",
                "PILOTAGE_ALLOWED_SENDERS": "212600000000",
                "PATH": "safe-path",
            },
            clear=True,
        ):
            child = web._worker_environment()
        self.assertEqual(child["PATH"], "safe-path")
        self.assertNotIn("OPENAI_API_KEY", child)
        self.assertNotIn("PILOTAGE_ALLOWED_SENDERS", child)


class WebToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_runs_off_the_async_agent_loop(self):
        main_thread = threading.get_ident()
        observed = {}

        def fake_search(query, limit):
            observed["thread"] = threading.get_ident()
            observed["query"] = query
            observed["limit"] = limit
            return {"success": True, "data": {"web": []}}

        with mock.patch.object(web, "_search", side_effect=fake_search):
            result = await web.handle_web_search(
                {"query": "current exchange rate", "limit": 3},
                ToolContext(chat_id="chat", config=None),
            )

        self.assertNotEqual(observed["thread"], main_thread)
        self.assertEqual(observed["query"], "current exchange rate")
        self.assertEqual(observed["limit"], 3)
        self.assertTrue(json.loads(result)["success"])

    async def test_empty_query_is_rejected_without_starting_a_worker(self):
        with mock.patch.object(web, "_search") as search:
            result = await web.handle_web_search(
                {"query": "   "},
                ToolContext(chat_id="chat", config=None),
            )
        search.assert_not_called()
        self.assertIn("error", json.loads(result))

    async def test_registry_offers_and_dispatches_the_web_tool(self):
        registry = build_registry()
        definition = registry.get("web_search")
        self.assertIsNotNone(definition)
        self.assertEqual(definition.group, "web")
        self.assertEqual(definition.schema["parameters"]["required"], ["query"])

        with mock.patch.object(
            web,
            "_search",
            return_value={"success": True, "data": {"web": []}},
        ):
            result = await registry.dispatch(
                "web_search",
                '{"query": "Pilotage"}',
                ToolContext(chat_id="chat", config=None),
                allowed_groups=["web"],
            )
        self.assertTrue(json.loads(result)["success"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
