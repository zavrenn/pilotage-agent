"""Contract for the Hermes-derived Codex image-generation tool."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx

from pilotage.codex import auth
from pilotage.settings import ConfigError, Settings
from pilotage.tools import ToolContext, build_registry
from pilotage.tools import image


_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)
_PNG_BYTES = bytes.fromhex(_PNG_HEX)


def _b64_png() -> str:
    return base64.b64encode(_PNG_BYTES).decode("ascii")


def _credentials(token: str = "codex-token") -> auth.Credentials:
    return auth.Credentials(
        access_token=token,
        refresh_token="refresh-token",
        base_url="https://chatgpt.com/backend-api/codex",
        last_refresh="",
    )


def _context(root: Path, settings: Settings | None = None) -> ToolContext:
    config = SimpleNamespace(
        settings=settings
        or Settings(
            {
                "image_gen": {
                    "provider": "openai-codex",
                    "model": "gpt-image-2-high",
                }
            }
        ),
        credentials_path=root / "codex-auth.json",
        main_credentials_path=root / "main-auth.json",
        workspace_dir=root / "workspace",
    )
    return ToolContext(chat_id="chat", config=config)


class ConfigurationTests(unittest.TestCase):
    def test_production_defaults_are_the_fixed_codex_high_tier(self):
        self.assertEqual(image.validate_image_settings(Settings()), "gpt-image-2-high")

    def test_another_provider_is_refused_not_silently_ignored(self):
        with self.assertRaisesRegex(ConfigError, "image_gen.provider"):
            image.validate_image_settings(
                Settings({"image_gen": {"provider": "fal"}})
            )

    def test_an_unknown_tier_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "image_gen.model"):
            image.validate_image_settings(
                Settings({"image_gen": {"model": "gpt-image-latest"}})
            )


class RequestShapeTests(unittest.TestCase):
    def test_payload_is_hermes_codex_hosted_tool_shape(self):
        payload = image._build_responses_payload(
            prompt="a red circle",
            size="1024x1536",
            quality="high",
        )

        self.assertEqual(payload["model"], "gpt-5.5")
        self.assertFalse(payload["store"])
        self.assertTrue(payload["stream"])
        self.assertNotIn("tool_choice", payload)
        self.assertEqual(payload["input"][0]["type"], "message")
        self.assertEqual(payload["input"][0]["role"], "user")
        self.assertEqual(
            payload["input"][0]["content"][0],
            {"type": "input_text", "text": "a red circle"},
        )
        self.assertEqual(
            payload["tools"][0],
            {
                "type": "image_generation",
                "model": "gpt-image-2",
                "size": "1024x1536",
                "quality": "high",
                "output_format": "png",
                "background": "opaque",
                "partial_images": 0,
            },
        )

    def test_input_images_follow_the_prompt(self):
        inputs = [{"type": "input_image", "image_url": "https://example.com/a.png"}]
        payload = image._build_responses_payload(
            prompt="edit it",
            size="1024x1024",
            quality="high",
            input_images=inputs,
        )
        self.assertEqual(payload["input"][0]["content"][1], inputs[0])

    def test_error_summary_prefers_the_actionable_message(self):
        body = json.dumps(
            {
                "metadata": "x" * 1_000,
                "error": {"message": "the useful diagnosis"},
            }
        )
        self.assertEqual(image._summarize_error_body(body), "the useful diagnosis")

    def test_non_json_error_body_is_bounded(self):
        self.assertEqual(len(image._summarize_error_body("x" * 1_000)), 500)


class SSETests(unittest.TestCase):
    def test_event_and_data_lines_are_combined(self):
        response = SimpleNamespace(
            iter_lines=lambda: iter(
                [
                    "event: response.output_item.done",
                    'data: {"item":{"type":"image_generation_call","result":"abc"}}',
                    "",
                ]
            )
        )
        events = list(image._iter_sse_json(response))
        self.assertEqual(events[0]["type"], "response.output_item.done")
        self.assertEqual(events[0]["item"]["result"], "abc")

    def test_partial_image_is_usable_when_done_is_missing(self):
        payload = {
            "type": "response.image_generation_call.partial_image",
            "partial_image_b64": "partial",
        }
        self.assertEqual(image._extract_image_b64(payload), "partial")
        final, partial = image._extract_image_candidates(payload)
        self.assertIsNone(final)
        self.assertEqual(partial, "partial")

    def test_final_image_wins_over_a_coexisting_partial(self):
        payload = {
            "type": "response.output_item.done",
            "item": {
                "type": "image_generation_call",
                "result": "final",
                "partial_image_b64": "partial",
            },
        }
        self.assertEqual(image._extract_image_b64(payload), "final")

    def test_nested_final_image_wins_over_a_sibling_partial(self):
        payload = {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "image_generation_call",
                        "status": "completed",
                        "result": "final",
                    }
                ]
            },
            "partial_image_b64": "partial",
        }
        self.assertEqual(image._extract_image_b64(payload), "final")

    def test_completed_response_is_scanned_recursively(self):
        payload = {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "image_generation_call",
                        "status": "completed",
                        "result": "final",
                    }
                ]
            },
        }
        self.assertEqual(image._extract_image_b64(payload), "final")

    def test_collect_uses_codex_headers_endpoint_and_stream(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                return iter(
                    [
                        "event: response.output_item.done",
                        'data: {"item":{"type":"image_generation_call","result":"abc"}}',
                        "",
                    ]
                )

        class FakeClient:
            def __init__(self, **kwargs):
                captured["client"] = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def stream(self, method, endpoint, **kwargs):
                captured["method"] = method
                captured["endpoint"] = endpoint
                captured["request"] = kwargs
                return FakeResponse()

        with mock.patch.object(image.httpx, "Client", FakeClient):
            result = image._collect_image_b64(
                _credentials(),
                prompt="a cat",
                size="1024x1024",
                quality="high",
            )

        self.assertEqual(result, {"b64": "abc", "source": "final"})
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(
            captured["endpoint"],
            "https://chatgpt.com/backend-api/codex/responses",
        )
        headers = captured["client"]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer codex-token")
        self.assertEqual(headers["originator"], "codex_cli_rs")
        self.assertIn("User-Agent", headers)
        self.assertNotIn("tool_choice", captured["request"]["json"])

    def test_http_error_surfaces_bounded_wire_message(self):
        body = json.dumps(
            {
                "metadata": "x" * 1_000,
                "error": {
                    "message": (
                        "Tool choice 'image_generation' not found in "
                        "'tools' parameter."
                    )
                },
            }
        )

        def handler(request):
            return httpx.Response(400, text=body, request=request)

        real_client = httpx.Client

        def client(*_args, **kwargs):
            return real_client(
                transport=httpx.MockTransport(handler),
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            )

        with (
            mock.patch.object(image.httpx, "Client", client),
            self.assertRaises(image.ImageAPIError) as caught,
        ):
            image._collect_image_b64(
                _credentials(),
                prompt="a cat",
                size="1024x1024",
                quality="high",
            )

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("HTTP 400", str(caught.exception))
        self.assertIn("tools' parameter", str(caught.exception))
        self.assertLess(len(str(caught.exception)), len(body))


class InputImageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_local_image_is_sniffed_and_encoded(self):
        source = self.root / "source.bin"
        source.write_bytes(_PNG_BYTES)
        part = image._to_input_image_part(str(source))
        self.assertTrue(part["image_url"].startswith("data:image/png;base64,"))

    def test_non_image_local_source_is_rejected(self):
        source = self.root / "not-image.png"
        source.write_text("not an image", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not a supported image"):
            image._to_input_image_part(str(source))

    def test_data_url_is_canonicalized_from_magic_bytes(self):
        written = f"data:image/jpeg;base64,{_b64_png()}"
        canonical = image._data_url_to_input_image_url(written)
        self.assertTrue(canonical.startswith("data:image/png;base64,"))

    def test_bad_data_url_is_rejected(self):
        with self.assertRaises(ValueError):
            image._data_url_to_input_image_url(
                "data:image/png;base64,bm90IGFuIGltYWdl"
            )

    def test_primary_and_references_preserve_order_and_cap(self):
        values = [f"https://example.com/{index}.png" for index in range(20)]
        parts = image._normalize_input_images(values[0], values[1:])
        self.assertEqual(len(parts), 16)
        self.assertEqual(parts[0]["image_url"], values[0])
        self.assertEqual(parts[-1]["image_url"], values[15])


class GenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.context = _context(self.root)

    def test_success_saves_a_deliverable_workspace_image(self):
        captured = {}

        def collect(credentials, **kwargs):
            captured["credentials"] = credentials
            captured.update(kwargs)
            return {"b64": _b64_png(), "source": "final"}

        with (
            mock.patch.object(
                image, "_resolve_credentials", return_value=_credentials()
            ),
            mock.patch.object(image, "_collect_image_b64", side_effect=collect),
        ):
            result = image._generate(
                {"prompt": "a cat", "aspect_ratio": "portrait"},
                self.context,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["provider"], "openai-codex")
        self.assertEqual(result["model"], "gpt-image-2-high")
        self.assertEqual(result["quality"], "high")
        self.assertEqual(result["size"], "1024x1536")
        self.assertEqual(result["modality"], "text")
        self.assertEqual(result["image_source"], "final")
        self.assertEqual(result["pixel_size"], "1x1")
        saved = Path(result["image"])
        self.assertTrue(saved.is_file())
        self.assertEqual(saved.read_bytes(), _PNG_BYTES)
        self.assertEqual(saved.parent, self.context.config.workspace_dir / "generated-images")
        self.assertEqual(captured["quality"], "high")

    def test_redirected_output_directory_is_refused(self):
        with mock.patch.object(
            image,
            "validate_within_dir",
            return_value="Path escapes allowed directory",
        ):
            with self.assertRaisesRegex(ValueError, "escapes allowed directory"):
                image._save_b64_image(
                    _b64_png(),
                    self.context.config.workspace_dir,
                    "gpt-image-2-high",
                )

        output_dir = self.context.config.workspace_dir / "generated-images"
        self.assertEqual(list(output_dir.iterdir()), [])

    def test_editing_forwards_local_source_pixels(self):
        source = self.root / "source.png"
        source.write_bytes(_PNG_BYTES)
        captured = {}

        def collect(_credentials, **kwargs):
            captured.update(kwargs)
            return {"b64": _b64_png(), "source": "final"}

        with (
            mock.patch.object(
                image, "_resolve_credentials", return_value=_credentials()
            ),
            mock.patch.object(image, "_collect_image_b64", side_effect=collect),
        ):
            result = image._generate(
                {"prompt": "make it blue", "image_url": str(source)},
                self.context,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["modality"], "image")
        self.assertEqual(result["input_image_count"], 1)
        self.assertTrue(
            captured["input_images"][0]["image_url"].startswith(
                "data:image/png;base64,"
            )
        )

    def test_missing_credentials_are_a_visible_auth_failure(self):
        with mock.patch.object(
            image,
            "_resolve_credentials",
            side_effect=auth.AuthError(
                "Run pilotage login",
                relogin_required=True,
            ),
        ):
            result = image._generate({"prompt": "a cat"}, self.context)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "auth_required")

    def test_invalid_source_is_returned_as_data(self):
        source = self.root / "bad.png"
        source.write_text("bad", encoding="utf-8")
        with mock.patch.object(
            image, "_resolve_credentials", return_value=_credentials()
        ):
            result = image._generate(
                {"prompt": "edit", "image_url": str(source)},
                self.context,
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "invalid_image_input")

    def test_empty_provider_response_is_not_success(self):
        with (
            mock.patch.object(
                image, "_resolve_credentials", return_value=_credentials()
            ),
            mock.patch.object(
                image, "_collect_image_b64", return_value=None
            ) as collect,
        ):
            result = image._generate({"prompt": "a cat"}, self.context)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "empty_response")
        self.assertEqual(collect.call_count, 2)

    def test_partial_only_response_retries_then_fails_closed(self):
        with (
            mock.patch.object(
                image, "_resolve_credentials", return_value=_credentials()
            ),
            mock.patch.object(
                image,
                "_collect_image_b64",
                return_value={"b64": _b64_png(), "source": "partial"},
            ) as collect,
        ):
            with self.assertLogs("pilotage.tools.image", level="WARNING"):
                result = image._generate({"prompt": "a cat"}, self.context)

        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "incomplete_image")
        self.assertEqual(result["image_source"], "partial")
        self.assertEqual(collect.call_count, 2)
        self.assertFalse(self.context.config.workspace_dir.exists())

    def test_partial_then_final_retry_succeeds(self):
        with (
            mock.patch.object(
                image, "_resolve_credentials", return_value=_credentials()
            ),
            mock.patch.object(
                image,
                "_collect_image_b64",
                side_effect=[
                    {"b64": _b64_png(), "source": "partial"},
                    {"b64": _b64_png(), "source": "final"},
                ],
            ) as collect,
        ):
            with self.assertLogs("pilotage.tools.image", level="WARNING"):
                result = image._generate({"prompt": "a cat"}, self.context)

        self.assertTrue(result["success"])
        self.assertEqual(result["image_source"], "final")
        self.assertEqual(collect.call_count, 2)

    def test_unauthorized_response_refreshes_once(self):
        first = _credentials("old-token")
        refreshed = _credentials("new-token")
        with (
            mock.patch.object(
                image,
                "_resolve_credentials",
                side_effect=[first, refreshed],
            ) as resolve,
            mock.patch.object(
                image,
                "_collect_image_b64",
                side_effect=[
                    image.ImageAPIError(401, "expired"),
                    {"b64": _b64_png(), "source": "final"},
                ],
            ),
        ):
            result = image._generate({"prompt": "a cat"}, self.context)

        self.assertTrue(result["success"])
        self.assertEqual(resolve.call_count, 2)
        self.assertEqual(
            resolve.call_args_list[-1],
            mock.call(self.context.config, force_refresh=True),
        )

    def test_non_auth_api_failure_is_not_retried(self):
        with (
            mock.patch.object(
                image, "_resolve_credentials", return_value=_credentials()
            ) as resolve,
            mock.patch.object(
                image,
                "_collect_image_b64",
                side_effect=image.ImageAPIError(429, "rate limited"),
            ),
        ):
            result = image._generate({"prompt": "a cat"}, self.context)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "api_error")
        self.assertIn("rate limited", result["error"])
        resolve.assert_called_once()


class ImageToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_runs_off_the_agent_event_loop(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        main_thread = threading.get_ident()
        observed = {}

        def generate(args, context):
            observed["thread"] = threading.get_ident()
            observed["args"] = args
            return {"success": True, "image": "x"}

        with mock.patch.object(image, "_generate_with_slot", side_effect=generate):
            result = await image.handle_image_generate(
                {"prompt": "a cat"},
                _context(Path(temporary.name)),
            )

        self.assertNotEqual(observed["thread"], main_thread)
        self.assertEqual(observed["args"]["prompt"], "a cat")
        self.assertTrue(json.loads(result)["success"])

    async def test_empty_prompt_is_rejected_without_starting_work(self):
        with mock.patch.object(image, "_generate_with_slot") as generate:
            result = await image.handle_image_generate(
                {"prompt": "  "},
                ToolContext(chat_id="chat", config=None),
            )
        generate.assert_not_called()
        self.assertIn("error", json.loads(result))

    async def test_registry_offers_and_dispatches_image_generation(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        registry = build_registry()
        tool = registry.get("image_generate")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.group, "image_gen")

        with mock.patch.object(
            image,
            "_generate_with_slot",
            return_value={"success": True, "image": "x"},
        ):
            result = await registry.dispatch(
                "image_generate",
                '{"prompt":"a cat"}',
                _context(Path(temporary.name)),
                allowed_groups=["image_gen"],
            )
        self.assertTrue(json.loads(result)["success"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
