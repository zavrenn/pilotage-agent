"""Astra selection with fake account catalogs and disposable policy files."""

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from pilotage import main, model_setup
from pilotage.codex import auth, models
from pilotage.config import Config
from pilotage.settings import ConfigError, Settings, set_agent_model


def catalog(levels=models.EFFORTS, **overrides):
    return {"models": [{
        "slug": models.MODEL, "visibility": "list", "supported_in_api": False,
        "supported_reasoning_levels": [{"effort": e} for e in levels], **overrides,
    }]}


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("unused-agent-auth.json")
        credentials = auth.Credentials("fake-access", "fake-refresh", auth.DEFAULT_CODEX_BASE_URL, "now")
        patch = mock.patch.object(auth, "resolve_credentials", return_value=credentials)
        self.resolve = patch.start()
        self.addCleanup(patch.stop)

    def test_only_native_astra_efforts_are_offered_from_this_account(self):
        with mock.patch.object(models.httpx, "get", return_value=httpx.Response(
            200, json=catalog(["ultra", "max", "low", "none", "medium"])
        )) as get:
            self.assertEqual(models.available_efforts(self.path), ("low", "medium", "max"))
        self.resolve.assert_called_once_with(self.path)
        self.assertEqual(get.call_args.args[0], auth.DEFAULT_CODEX_BASE_URL + "/models")
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer fake-access")
        self.assertEqual(get.call_args.kwargs["headers"]["originator"], "codex_cli_rs")

    def test_authentication_refresh_is_bounded(self):
        with mock.patch.object(models.httpx, "get", side_effect=[
            httpx.Response(401), httpx.Response(200, json=catalog())
        ]):
            self.assertEqual(models.available_efforts(self.path), models.EFFORTS)
        self.resolve.assert_has_calls([
            mock.call(self.path), mock.call(self.path, force_refresh=True),
        ])

    def test_missing_hidden_or_malformed_astra_never_uses_another_model(self):
        for payload in (
            {}, [], {"models": []}, catalog(slug="gpt-5.6-sol"),
            catalog(visibility="hidden"), catalog(visibility="none"),
            catalog(show_in_picker=False), catalog(supported_reasoning_levels=None),
            catalog(["ultra", "none"]),
        ):
            with self.subTest(payload=payload), mock.patch.object(models.httpx, "get", return_value=httpx.Response(200, json=payload)):
                with self.assertRaises(models.ModelError):
                    models.available_efforts(self.path)

    def test_unavailable_catalog_fails_without_displaying_response_secrets(self):
        for response in (httpx.Response(403, text="private response"), httpx.Response(200, text="invalid")):
            with mock.patch.object(models.httpx, "get", return_value=response), self.assertRaises(models.ModelError) as error:
                models.available_efforts(self.path)
            self.assertNotIn("private response", str(error.exception))

    def test_catalog_effort_strings_are_supported(self):
        with mock.patch.object(models.httpx, "get", return_value=httpx.Response(
            200, json=catalog(supported_reasoning_levels=["low", "high", "ultra"])
        )):
            self.assertEqual(models.available_efforts(self.path), ("low", "high"))


class SetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "config.yaml"
        self.original = "# policy\nagent:\n  model: gpt-5.6-sol # selected\n  reasoning_effort: medium\n  history_turns: 12\ntools:\n  enabled: []\n"
        self.path.write_text(self.original, encoding="utf-8")

    def run_setup(self, choices, *, efforts=models.EFFORTS):
        self.output = io.StringIO()
        with (
            mock.patch.object(model_setup, "_available_efforts", return_value=efforts),
            mock.patch.object(model_setup.sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", side_effect=choices),
            contextlib.redirect_stdout(self.output), contextlib.redirect_stderr(self.output),
        ):
            return model_setup.run_model_setup(self.root, self.path)

    def test_picker_validates_choices_and_saves_only_selected_defaults(self):
        self.assertEqual(self.run_setup(["high", "", "99", "2"], efforts=("low", "high")), 0)
        expected = self.original.replace("gpt-5.6-sol", models.MODEL).replace("reasoning_effort: medium", "reasoning_effort: high")
        self.assertEqual(self.path.read_text(encoding="utf-8"), expected)
        self.assertIn("pilotage restart", self.output.getvalue())

    def test_cancel_and_eof_never_change_configuration(self):
        for choices in (["0"], [EOFError()], [KeyboardInterrupt()]):
            self.assertEqual(self.run_setup(choices), 0)
            self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)

    def test_high_is_the_initial_default_in_config_template_and_picker(self):
        self.path.write_text("tools:\n  enabled: []\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.root)}, clear=True):
            self.assertEqual(Config.load().reasoning_effort, "high")
        template = Settings.load(Path(__file__).parents[1] / "config.yaml.example")
        self.assertEqual(template.text("agent.reasoning_effort"), "high")
        self.assertEqual(self.run_setup(["0"]), 0)
        self.assertIn("Current default effort: high", self.output.getvalue())
        self.assertIn("high (current)", self.output.getvalue())

    def test_noninteractive_invocation_does_not_probe_or_write(self):
        with mock.patch.object(model_setup.sys.stdin, "isatty", return_value=False), mock.patch.object(model_setup, "_available_efforts") as probe, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(model_setup.run_model_setup(self.root, self.path), 1)
        probe.assert_not_called()
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)

    def test_failed_availability_does_not_save(self):
        with mock.patch.object(model_setup.sys.stdin, "isatty", return_value=True), mock.patch.object(model_setup, "_available_efforts", side_effect=models.ModelError("not available")), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(model_setup.run_model_setup(self.root, self.path), 1)
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)

    def test_defaults_can_be_added_to_small_policy_files(self):
        for original in ("", "# comment\n", "{}", "agent: {}\n", "agent:\n  history_turns: 12\n", "tools:\n  enabled: []\n"):
            with self.subTest(original=original):
                self.path.write_text(original, encoding="utf-8")
                set_agent_model(self.path, models.MODEL, "max")
                saved = Settings.load(self.path)
                self.assertEqual(saved.text("agent.model"), models.MODEL)
                self.assertEqual(saved.text("agent.reasoning_effort"), "max")

    def test_failed_replace_keeps_original_policy(self):
        with mock.patch("pilotage.settings.os.replace", side_effect=OSError("failed")), self.assertRaises(OSError):
            set_agent_model(self.path, models.MODEL, "high")
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_ambiguous_yaml_is_not_rewritten(self):
        for original in ("agent: [broken\n", "agent: {}\nagent: {}\n", "agent: &shared {model: old}\n", "[]", "false", "agent: null\n"):
            with self.subTest(original=original):
                self.path.write_text(original, encoding="utf-8")
                with self.assertRaises(ConfigError):
                    set_agent_model(self.path, models.MODEL, "high")
                self.assertEqual(self.path.read_text(encoding="utf-8"), original)

    def test_protected_setup_probes_as_agent_without_reading_its_credentials(self):
        with (
            mock.patch.object(model_setup.deployment, "load", return_value=object()),
            mock.patch.object(model_setup.deployment, "agent_python", return_value=["fake-agent-python"]) as command,
            mock.patch.object(model_setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(["low", "high"]), "")) as run,
            mock.patch.object(models, "available_efforts", side_effect=AssertionError("operator read auth")),
        ):
            self.assertEqual(model_setup._available_efforts(self.root), ("low", "high"))
        command.assert_called_once_with(["-m", "pilotage.codex.models"])
        self.assertEqual(run.call_args.args, (["fake-agent-python"],))

    def test_model_command_works_before_channel_setup_and_rejects_flags(self):
        with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.root)}, clear=True), mock.patch.object(main, "load_env_files", return_value=[]), mock.patch.object(main.Config, "load", side_effect=AssertionError("strict config")), mock.patch.object(model_setup, "run_model_setup", return_value=0) as setup:
            self.assertEqual(main.main(["model"]), 0)
        setup.assert_called_once_with(self.root, self.path)
        for args in (["model", "gpt-6-astra"], ["model", "--effort", "high"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_astra_configuration_rejects_unsupported_efforts_and_models(self):
        with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.root)}, clear=True):
            for value in models.EFFORTS:
                self.path.write_text(f"agent:\n  reasoning_effort: {value}\n", encoding="utf-8")
                self.assertEqual(Config.load().reasoning_effort, value)
            for value in ("none", "minimal", "ultra", "default", "random"):
                self.path.write_text(f"agent:\n  reasoning_effort: {value}\n", encoding="utf-8")
                with self.assertRaisesRegex(ConfigError, "reasoning_effort"):
                    Config.load()
            for value in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
                self.path.write_text(f"agent:\n  model: {value}\n", encoding="utf-8")
                with self.assertRaisesRegex(ConfigError, "agent.model"):
                    Config.load()
