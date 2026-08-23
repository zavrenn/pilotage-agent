"""Contract for the Hermes-derived DDGS web-search tool."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from pilotage.settings import ConfigError, Settings
from pilotage.tools import ToolContext, build_registry
from pilotage.tools import web


def _tool_context(root: Path, settings=None, result_limit=100_000) -> ToolContext:
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config = types.SimpleNamespace(
        state_dir=root,
        workspace_dir=workspace,
        settings=Settings(settings or {}),
        max_tool_result_chars=result_limit,
    )
    return ToolContext(chat_id="chat", config=config)


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


class WebExtractSettingsTests(unittest.TestCase):
    def test_extract_budget_is_validated_at_startup(self):
        self.assertEqual(web.validate_web_settings(Settings()), 15_000)
        with self.assertRaisesRegex(ConfigError, "between 2000 and 500000"):
            web.validate_web_settings(
                Settings({"web": {"extract_char_limit": 1_999}})
            )

    def test_firecrawl_requires_direct_cloud_or_self_hosted_config(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "FIRECRAWL_API_KEY"):
                web._get_direct_firecrawl_config()

    def test_pinned_firecrawl_sdk_exposes_hermes_scrape_surface(self):
        original_client = web._firecrawl_client
        original_config = web._firecrawl_client_config
        self.addCleanup(setattr, web, "_firecrawl_client", original_client)
        self.addCleanup(
            setattr,
            web,
            "_firecrawl_client_config",
            original_config,
        )
        web._firecrawl_client = None
        web._firecrawl_client_config = None
        with mock.patch.dict(
            os.environ,
            {"FIRECRAWL_API_KEY": "fc-test-placeholder"},
            clear=True,
        ):
            client = web._get_firecrawl_client()
        self.assertTrue(callable(client.scrape))


class FirecrawlExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_scrape_runs_off_loop_and_normalizes_nested_payload(self):
        main_thread = threading.get_ident()
        observed = {}

        class Client:
            def scrape(self, *, url, formats):
                observed.update(
                    thread=threading.get_ident(),
                    url=url,
                    formats=formats,
                )
                return {
                    "data": {
                        "markdown": "clean markdown",
                        "html": "<p>clean html</p>",
                        "metadata": {
                            "title": "Page",
                            "sourceURL": "https://example.com/final",
                        },
                    }
                }

        with (
            mock.patch.object(web, "_get_firecrawl_client", return_value=Client()),
            mock.patch.object(
                web,
                "async_is_safe_url",
                new=mock.AsyncMock(return_value=True),
            ) as safe,
        ):
            results = await web._extract_firecrawl(
                ["https://example.com/start"],
                format="markdown",
            )

        self.assertNotEqual(observed["thread"], main_thread)
        self.assertEqual(observed["formats"], ["markdown"])
        self.assertEqual(results[0]["title"], "Page")
        self.assertEqual(results[0]["content"], "clean markdown")
        self.assertEqual(results[0]["url"], "https://example.com/final")
        safe.assert_awaited_once_with("https://example.com/final")

    async def test_per_url_timeout_is_returned_without_raising(self):
        class SlowClient:
            def scrape(self, **_kwargs):
                time.sleep(0.05)
                return {}

        with (
            mock.patch.object(
                web,
                "_get_firecrawl_client",
                return_value=SlowClient(),
            ),
            mock.patch.object(web, "_EXTRACT_TIMEOUT_SECONDS", 0.01),
        ):
            results = await web._extract_firecrawl(["https://example.com"])

        self.assertIn("timed out", results[0]["error"])

    async def test_unsafe_reported_redirect_is_blocked(self):
        class Client:
            def scrape(self, **_kwargs):
                return {
                    "markdown": "private",
                    "metadata": {
                        "source_url": "http://127.0.0.1/private",
                    },
                }

        safe = mock.AsyncMock(return_value=False)
        with (
            mock.patch.object(web, "_get_firecrawl_client", return_value=Client()),
            mock.patch.object(
                web,
                "async_is_safe_url",
                new=safe,
            ),
        ):
            results = await web._extract_firecrawl(["https://example.com"])

        safe.assert_awaited_once_with("http://127.0.0.1/private")
        self.assertIn("private or internal", results[0]["error"])
        self.assertEqual(results[0]["content"], "")

    async def test_credential_bearing_reported_redirect_is_not_exposed(self):
        class Client:
            def scrape(self, **_kwargs):
                return {
                    "markdown": "private",
                    "metadata": {
                        "source_url": "https://alice:hunter2@example.com/final",
                    },
                }

        safe = mock.AsyncMock(return_value=True)
        with (
            mock.patch.object(web, "_get_firecrawl_client", return_value=Client()),
            mock.patch.object(web, "async_is_safe_url", new=safe),
        ):
            results = await web._extract_firecrawl(["https://example.com/start"])

        safe.assert_not_awaited()
        self.assertIn("username/password", results[0]["error"])
        self.assertEqual(results[0]["url"], "https://example.com/start")
        self.assertNotIn("hunter2", json.dumps(results))


class WebExtractHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_known_secret_is_blocked_before_any_fetch(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            context = _tool_context(Path(raw_root))
            with (
                mock.patch.object(
                    web,
                    "async_is_safe_url",
                    new=mock.AsyncMock(),
                ) as safe,
                mock.patch.object(
                    web,
                    "_extract_firecrawl",
                    new=mock.AsyncMock(),
                ) as extract,
            ):
                result = await web.handle_web_extract(
                    {"urls": ["https://example.com/%73k-abcdefghij"]},
                    context,
                )

        self.assertFalse(json.loads(result)["success"])
        self.assertIn("API key or token", json.loads(result)["error"])
        safe.assert_not_awaited()
        extract.assert_not_awaited()

    async def test_sensitive_query_parameter_is_blocked_before_fetch(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            context = _tool_context(Path(raw_root))
            with mock.patch.object(
                web,
                "_extract_firecrawl",
                new=mock.AsyncMock(),
            ) as extract:
                result = await web.handle_web_extract(
                    {"urls": ["https://example.com/page?token=opaque"]},
                    context,
                )

        self.assertFalse(json.loads(result)["success"])
        self.assertIn("credential-like query", json.loads(result)["error"])
        extract.assert_not_awaited()

    async def test_url_userinfo_credentials_are_blocked_before_fetch(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            context = _tool_context(Path(raw_root))
            with (
                mock.patch.object(
                    web,
                    "async_is_safe_url",
                    new=mock.AsyncMock(),
                ) as safe,
                mock.patch.object(
                    web,
                    "_extract_firecrawl",
                    new=mock.AsyncMock(),
                ) as extract,
            ):
                for candidate in (
                    "https://alice:hunter2@example.com/page",
                    "https://alice%3Ahunter2@example.com/page",
                ):
                    with self.subTest(candidate=candidate):
                        result = await web.handle_web_extract(
                            {"urls": [candidate]},
                            context,
                        )
                        self.assertFalse(json.loads(result)["success"])
                        self.assertIn("username/password", result)
                        self.assertNotIn("hunter2", result)

        safe.assert_not_awaited()
        extract.assert_not_awaited()

    async def test_input_order_survives_invalid_and_ssrf_blocked_items(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            context = _tool_context(Path(raw_root))
            safe = mock.AsyncMock(side_effect=[True, False])
            extract = mock.AsyncMock(
                return_value=[
                    {
                        "url": "https://xn--exmple-cua.com/a%20b",
                        "title": "Public",
                        "content": "body",
                    }
                ]
            )
            with (
                mock.patch.object(web, "async_is_safe_url", new=safe),
                mock.patch.object(web, "_extract_firecrawl", new=extract),
            ):
                result = await web.handle_web_extract(
                    {
                        "urls": [
                            {"href": "https://exämple.com/a b"},
                            7,
                            "http://127.0.0.1/private",
                        ]
                    },
                    context,
                )

        payload = json.loads(result)
        self.assertEqual(len(payload["results"]), 3)
        self.assertEqual(payload["results"][0]["title"], "Public")
        self.assertIn("Invalid URL item at index 1", payload["results"][1]["error"])
        self.assertIn("private or internal", payload["results"][2]["error"])
        extract.assert_awaited_once_with(
            ["https://xn--exmple-cua.com/a%20b"],
            format="markdown",
        )

    async def test_long_page_is_cleaned_truncated_and_saved_per_profile(self):
        inline_image = "![chart](data:image/png;base64,AAAA)"
        long_content = inline_image + "\n" + ("line of useful text\n" * 180)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            root = Path(raw_root)
            context = _tool_context(root)
            extract = mock.AsyncMock(
                return_value=[
                    {
                        "url": "https://example.com/long",
                        "title": "Long",
                        "content": long_content,
                        "raw_content": long_content,
                    }
                ]
            )
            with (
                mock.patch.object(
                    web,
                    "async_is_safe_url",
                    new=mock.AsyncMock(return_value=True),
                ),
                mock.patch.object(web, "_extract_firecrawl", new=extract),
            ):
                result = await web.handle_web_extract(
                    {
                        "urls": ["https://example.com/long"],
                        "char_limit": 2_000,
                    },
                    context,
                )

            content = json.loads(result)["results"][0]["content"]
            stored = list((root / "cache" / "web").glob("*.md"))
            self.assertEqual(len(stored), 1)
            stored_text = stored[0].read_text(encoding="utf-8")
            if sys.platform != "win32":
                read_result = await build_registry().dispatch(
                    "read_file",
                    json.dumps({"path": str(stored[0]), "limit": 20}),
                    context,
                    allowed_groups=["file"],
                )
                read_payload = json.loads(read_result)
                self.assertIn("content", read_payload, read_payload)
                self.assertIn("[IMAGE: chart]", read_payload["content"])

        self.assertIn("[IMAGE: chart]", content)
        self.assertNotIn("base64", content)
        self.assertIn("[TRUNCATED]", content)
        self.assertIn("read_file path=", content)
        self.assertIn("[IMAGE: chart]", stored_text)
        self.assertNotIn("base64", stored_text)

    async def test_backend_short_result_gets_explicit_error(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            context = _tool_context(Path(raw_root))
            with (
                mock.patch.object(
                    web,
                    "async_is_safe_url",
                    new=mock.AsyncMock(return_value=True),
                ),
                mock.patch.object(
                    web,
                    "_extract_firecrawl",
                    new=mock.AsyncMock(return_value=[]),
                ),
            ):
                result = await web.handle_web_extract(
                    {"urls": ["https://example.com"]},
                    context,
                )

        self.assertIn(
            "returned no result",
            json.loads(result)["results"][0]["error"],
        )

    async def test_combined_previews_fit_runtime_cap_and_keep_cache_footers(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            root = Path(raw_root)
            context = _tool_context(root, result_limit=100_000)
            provider_results = [
                {
                    "url": f"https://example.com/page-{index}",
                    "title": f"Page {index}",
                    "content": ('line "quoted"\n' * 3_000),
                }
                for index in range(5)
            ]
            with (
                mock.patch.object(
                    web,
                    "async_is_safe_url",
                    new=mock.AsyncMock(return_value=True),
                ),
                mock.patch.object(
                    web,
                    "_extract_firecrawl",
                    new=mock.AsyncMock(return_value=provider_results),
                ),
            ):
                result = await web.handle_web_extract(
                    {
                        "urls": [item["url"] for item in provider_results],
                        "char_limit": 500_000,
                    },
                    context,
                )

            payload = json.loads(result)
            self.assertLessEqual(len(result), context.config.max_tool_result_chars)
            self.assertNotIn("truncated", payload)
            self.assertEqual(len(payload["results"]), 5)
            for item in payload["results"]:
                self.assertIn("[TRUNCATED]", item["content"])
                self.assertIn("read_file path=", item["content"])
            self.assertEqual(
                len(list((root / "cache" / "web").glob("*.md"))),
                5,
            )

    async def test_registry_offers_and_dispatches_web_extract(self):
        registry = build_registry()
        definition = registry.get("web_extract")
        self.assertIsNotNone(definition)
        self.assertEqual(definition.group, "web")
        self.assertEqual(definition.schema["parameters"]["required"], ["urls"])

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            context = _tool_context(Path(raw_root))
            with (
                mock.patch.object(
                    web,
                    "async_is_safe_url",
                    new=mock.AsyncMock(return_value=True),
                ),
                mock.patch.object(
                    web,
                    "_extract_firecrawl",
                    new=mock.AsyncMock(
                        return_value=[
                            {
                                "url": "https://example.com",
                                "title": "Example",
                                "content": "body",
                            }
                        ]
                    ),
                ),
            ):
                result = await registry.dispatch(
                    "web_extract",
                    '{"urls":["https://example.com"]}',
                    context,
                    allowed_groups=["web"],
                )

        self.assertEqual(json.loads(result)["results"][0]["content"], "body")


class WebExtractSpillTests(unittest.TestCase):
    def test_refreshing_cached_url_does_not_follow_preplanted_symlink(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw_root:
            root = Path(raw_root)
            context = _tool_context(root)
            first = web._store_full_text(
                context,
                "https://example.com/page",
                "first",
            )
            self.assertIsNotNone(first)
            cached = Path(first)
            cached.unlink()
            victim = root / "victim.txt"
            victim.write_text("untouched", encoding="utf-8")
            try:
                os.symlink(victim, cached)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            refreshed = web._store_full_text(
                context,
                "https://example.com/page",
                "second",
            )

            self.assertEqual(victim.read_text(encoding="utf-8"), "untouched")
            self.assertEqual(Path(refreshed).read_text(encoding="utf-8"), "second")
            self.assertFalse(Path(refreshed).is_symlink())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
