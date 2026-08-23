"""Native vision behavior ported from Hermes into the Codex-only loop."""

from __future__ import annotations

import asyncio
import base64
import copy
import importlib.util
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

from pilotage.agent import Agent
from pilotage.codex import stream as codex_stream
from pilotage.config import Config
from pilotage.history import ConversationStore
from pilotage.tools import (
    Registry,
    Tool,
    ToolContext,
    build_registry,
    responses_tool_output,
    run_calls,
)
from pilotage.tools import url_safety, vision


PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode(
    "ascii"
)


def _context(root: Path) -> ToolContext:
    config = SimpleNamespace(state_dir=root)
    return ToolContext(chat_id="chat", config=config)


def _call(call_id: str, name: str, **arguments: Any) -> Dict[str, str]:
    return {
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


class VisionSourceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.context = _context(self.root)

    async def test_data_url_becomes_hermes_multimodal_envelope(self):
        result = await vision.handle_vision_analyze(
            {
                "image_url": PNG_DATA_URL,
                "question": "What is in this image?",
            },
            self.context,
        )

        self.assertIs(result.get("_multimodal"), True)
        self.assertEqual(
            [part["type"] for part in result["content"]],
            ["text", "image_url"],
        )
        self.assertIn("What is in this image?", result["content"][0]["text"])
        embedded = result["content"][1]["image_url"]["url"]
        self.assertTrue(embedded.startswith("data:image/png;base64,"))
        self.assertTrue(result["meta"]["native_vision"])
        self.assertEqual(list((self.root / "cache" / "vision").iterdir()), [])

    async def test_relative_path_uses_the_profile_workspace(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        source = workspace / "photo.png"
        source.write_bytes(PNG_BYTES)

        result = await vision.handle_vision_analyze(
            {"image_url": "photo.png", "question": "Inspect it."},
            self.context,
        )

        self.assertIs(result.get("_multimodal"), True)
        self.assertEqual(result["meta"]["image_url"], "photo.png")

    async def test_file_uri_is_supported(self):
        source = self.root / "source.png"
        source.write_bytes(PNG_BYTES)

        result = await vision.handle_vision_analyze(
            {
                "image_url": source.resolve().as_uri(),
                "question": "Inspect it.",
            },
            self.context,
        )

        self.assertIs(result.get("_multimodal"), True)

    async def test_non_image_data_is_rejected(self):
        payload = base64.b64encode(b"not an image").decode("ascii")
        result = await vision.handle_vision_analyze(
            {
                "image_url": f"data:image/png;base64,{payload}",
                "question": "Inspect it.",
            },
            self.context,
        )

        self.assertIn("not a recognized image", json.loads(result)["error"])

    async def test_failed_format_conversion_leaves_no_temp_file(self):
        payload = base64.b64encode(b"BM" + b"\x00" * 32).decode("ascii")
        result = await vision.handle_vision_analyze(
            {
                "image_url": f"data:image/bmp;base64,{payload}",
                "question": "Inspect it.",
            },
            self.context,
        )

        self.assertIn("could not be converted", json.loads(result)["error"])
        self.assertEqual(
            list((self.root / "cache" / "vision").iterdir()),
            [],
        )

    async def test_secret_bearing_local_file_is_refused(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        (workspace / ".env").write_bytes(PNG_BYTES)

        result = await vision.handle_vision_analyze(
            {"image_url": ".env", "question": "Inspect it."},
            self.context,
        )

        self.assertIn("carries secrets", json.loads(result)["error"])

    async def test_unsupported_scheme_is_rejected(self):
        result = await vision.handle_vision_analyze(
            {"image_url": "ftp://example.com/a.png", "question": "Inspect it."},
            self.context,
        )
        self.assertIn("Unrecognized image source scheme", json.loads(result)["error"])

    async def test_http_download_uses_the_same_validation_pipeline(self):
        async def fake_download(_url: str, destination: Path) -> None:
            destination.write_bytes(PNG_BYTES)

        with mock.patch.object(
            vision,
            "_download_image",
            side_effect=fake_download,
        ) as download:
            result = await vision.handle_vision_analyze(
                {
                    "image_url": "https://example.com/photo",
                    "question": "Inspect it.",
                },
                self.context,
            )

        self.assertIs(result.get("_multimodal"), True)
        download.assert_awaited_once()
        self.assertEqual(list((self.root / "cache" / "vision").iterdir()), [])

    @unittest.skipUnless(
        importlib.util.find_spec("PIL"),
        "Pillow is declared for deployment but not installed in this test venv",
    )
    async def test_region_is_cropped_before_embedding(self):
        from PIL import Image

        workspace = self.root / "workspace"
        workspace.mkdir()
        source = workspace / "large.png"
        Image.new("RGB", (100, 80), "white").save(source)

        result = await vision.handle_vision_analyze(
            {
                "image_url": "large.png",
                "question": "Read the detail.",
                "region": [10, 20, 60, 70],
            },
            self.context,
        )

        self.assertIs(result.get("_multimodal"), True)
        note = result["content"][0]["text"]
        self.assertIn("crop starting at (10, 20)", note)


class MultimodalContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _envelope(self, image_data: str = PNG_DATA_URL) -> Dict[str, Any]:
        return {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": "Image loaded."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data},
                },
            ],
            "text_summary": "Image loaded.",
        }

    def test_codex_output_is_an_input_text_and_input_image_array(self):
        output = responses_tool_output(self._envelope())
        self.assertEqual(
            [part["type"] for part in output],
            ["input_text", "input_image"],
        )
        self.assertEqual(output[1]["image_url"], PNG_DATA_URL)

    async def test_registry_preserves_multimodal_results(self):
        registry = Registry()
        envelope = self._envelope()
        registry.register(
            Tool(
                name="vision",
                group="vision",
                schema={"name": "vision", "parameters": {"type": "object"}},
                handler=lambda _args, _context: envelope,
            )
        )
        result = await registry.dispatch(
            "vision",
            "{}",
            _context(self.root),
            allowed_groups=["vision"],
        )
        self.assertIs(result, envelope)

    async def test_step_budget_counts_summary_not_base64(self):
        registry = Registry()
        envelope = self._envelope(
            "data:image/png;base64," + "A" * 1_000_000
        )
        registry.register(
            Tool(
                name="vision",
                group="vision",
                schema={"name": "vision", "parameters": {"type": "object"}},
                handler=lambda _args, _context: envelope,
            )
        )
        results = await run_calls(
            registry,
            [_call("call_1", "vision")],
            _context(self.root),
            allowed_groups=["vision"],
            step_budget_chars=100,
        )
        self.assertIs(results[0], envelope)

    def test_watchdog_counts_tool_image_as_pixels_not_base64_text(self):
        request = {
            "instructions": "Inspect.",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": responses_tool_output(
                        self._envelope(
                            "data:image/png;base64," + "A" * 4_000_000
                        )
                    ),
                }
            ],
        }
        self.assertLess(codex_stream.estimate_context_tokens(request), 10_000)

    async def test_agent_keeps_one_follow_up_then_retires_image_bytes(self):
        with mock.patch.dict(
            "os.environ",
            {"PILOTAGE_HOME": str(self.root / "profile")},
        ):
            config = Config.load()
        agent = Agent(
            config,
            ConversationStore(self.root / "conversations.db"),
        )
        agent._registry._tools["vision_analyze"] = Tool(
            name="vision_analyze",
            group="vision",
            schema={
                "name": "vision_analyze",
                "parameters": {"type": "object"},
            },
            handler=lambda _args, _context: self._envelope(),
        )

        replies: List[codex_stream.StreamResult] = [
            codex_stream.StreamResult(
                tool_calls=[
                    _call(
                        "vision_1",
                        "vision_analyze",
                        image_url="photo.png",
                        question="Inspect it.",
                    )
                ]
            ),
            codex_stream.StreamResult(text="It is a photo."),
            codex_stream.StreamResult(text="The follow-up answer."),
            codex_stream.StreamResult(text="A later answer."),
        ]
        requests: List[Dict[str, Any]] = []

        async def stream_once(
            request: Dict[str, Any],
            *,
            force_refresh: bool,
            ttfb_timeout: float,
            idle_timeout: float,
        ) -> codex_stream.StreamResult:
            requests.append(copy.deepcopy(request))
            return replies.pop(0)

        agent._stream_once = stream_once

        await agent.respond("chat", "Inspect the photo.")
        tool_output = next(
            item
            for item in requests[1]["input"]
            if item.get("type") == "function_call_output"
        )
        self.assertIsInstance(tool_output["output"], list)

        await agent.respond("chat", "What color was it?")
        follow_up_output = next(
            item
            for item in requests[2]["input"]
            if item.get("type") == "function_call_output"
        )
        self.assertIsInstance(follow_up_output["output"], list)

        await agent.respond("chat", "Now something unrelated.")
        retired_output = next(
            item
            for item in requests[3]["input"]
            if item.get("type") == "function_call_output"
        )
        self.assertIsInstance(retired_output["output"], str)
        self.assertNotIn("base64", retired_output["output"])


class UrlSafetyTests(unittest.TestCase):
    def test_loopback_and_cloud_metadata_are_blocked(self):
        self.assertFalse(url_safety.is_safe_url("http://127.0.0.1/image.png"))
        self.assertFalse(
            url_safety.is_safe_url(
                "http://metadata.google.internal/latest/meta-data"
            )
        )

    def test_public_dns_answer_is_allowed(self):
        answer = [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 443),
            )
        ]
        with mock.patch("socket.getaddrinfo", return_value=answer):
            self.assertTrue(
                url_safety.is_safe_url("https://example.com/image.png")
            )

    def test_private_dns_answer_is_blocked(self):
        answer = [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("10.0.0.2", 443),
            )
        ]
        with mock.patch("socket.getaddrinfo", return_value=answer):
            self.assertFalse(
                url_safety.is_safe_url("https://example.com/image.png")
            )


class UrlTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_guarded_http_client_constructs_with_the_pinned_httpx(self):
        client = url_safety.create_ssrf_safe_async_client()
        try:
            backend = client._transport._pool._network_backend
            self.assertEqual(
                type(backend).__name__,
                "_SSRFGuardedAsyncNetworkBackend",
            )
        finally:
            await client.aclose()


class BuiltVisionTests(unittest.TestCase):
    def test_registry_exposes_only_the_production_vision_tool(self):
        registry = build_registry()
        self.assertEqual(registry.names(["vision"]), ["vision_analyze"])
        schema = registry.get("vision_analyze").schema
        self.assertEqual(
            schema["parameters"]["required"],
            ["image_url", "question"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
