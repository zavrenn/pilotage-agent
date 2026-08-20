"""The configuration file.

What matters here is not that YAML parses. It is that an operator who edits
this file gets exactly what they wrote: a missing file changes nothing, a
broken file stops the agent instead of quietly restoring a default, and a
setting written under a channel applies to that channel and to no other.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pilotage.config import Config
from pilotage.settings import ConfigError, Settings, config_path


class LoadingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.path = self.home / "config.yaml"

    def _write(self, text: str) -> Settings:
        self.path.write_text(text, encoding="utf-8")
        return Settings.load(self.path)

    def test_no_file_at_all_is_not_an_error(self):
        settings = Settings.load(self.home / "nothing.yaml")
        self.assertEqual(settings.text("agent.model", "default"), "default")

    def test_an_empty_file_is_not_an_error(self):
        settings = self._write("")
        self.assertEqual(settings.text("agent.model", "default"), "default")

    def test_a_broken_file_stops_the_agent(self):
        with self.assertRaises(ConfigError):
            self._write("agent:\n  model: [unclosed\n")

    def test_a_file_that_is_not_settings_stops_the_agent(self):
        with self.assertRaises(ConfigError):
            self._write("- one\n- two\n")

    def test_the_error_names_the_file(self):
        with self.assertRaises(ConfigError) as caught:
            self._write("agent:\n  model: [unclosed\n")
        self.assertIn(str(self.path), str(caught.exception))

    def test_a_nested_key_is_read_by_its_dotted_name(self):
        settings = self._write("agent:\n  model: gpt-test\n")
        self.assertEqual(settings.text("agent.model"), "gpt-test")

    def test_a_missing_key_leaves_the_default(self):
        settings = self._write("agent:\n  model: gpt-test\n")
        self.assertEqual(settings.text("agent.reasoning_effort", "medium"), "medium")

    def test_a_blank_value_is_a_missing_one(self):
        settings = self._write("agent:\n  model: '   '\n")
        self.assertEqual(settings.text("agent.model", "gpt-default"), "gpt-default")

    def test_a_wrong_type_is_refused_rather_than_coerced(self):
        settings = self._write("agent:\n  history_turns: many\n")
        with self.assertRaises(ConfigError):
            settings.count("agent.history_turns", 20)

    def test_a_fractional_count_is_not_silently_truncated(self):
        settings = self._write("agent:\n  history_turns: 2.5\n")
        with self.assertRaises(ConfigError):
            settings.count("agent.history_turns", 20)

    def test_a_boolean_is_not_a_number(self):
        settings = self._write("agent:\n  request_timeout: true\n")
        with self.assertRaises(ConfigError):
            settings.number("agent.request_timeout", 300.0)

    def test_a_non_finite_number_is_refused(self):
        settings = self._write("agent:\n  request_timeout: .inf\n")
        with self.assertRaises(ConfigError):
            settings.number("agent.request_timeout", 300.0)

    def test_structured_data_is_not_stringified_into_text(self):
        settings = self._write("agent:\n  model: [gpt, typo]\n")
        with self.assertRaises(ConfigError):
            settings.text("agent.model")

    def test_flags_accept_what_operators_actually_write(self):
        settings = self._write("a:\n  yaml: true\n  word: 'yes'\n  digit: 1\n  off: 'no'\n")
        self.assertTrue(settings.flag("a.yaml"))
        self.assertTrue(settings.flag("a.word"))
        self.assertTrue(settings.flag("a.digit"))
        self.assertFalse(settings.flag("a.off"))

    def test_an_unknown_flag_word_is_refused(self):
        settings = self._write("whatsapp:\n  answer_groups: yess\n")
        with self.assertRaises(ConfigError):
            settings.flag("whatsapp.answer_groups")

    def test_a_list_can_be_written_either_way(self):
        as_list = self._write("tools:\n  disabled: [terminal, file]\n")
        self.assertEqual(as_list.names("tools.disabled"), ["terminal", "file"])
        as_line = self._write("tools:\n  disabled: 'terminal, file'\n")
        self.assertEqual(as_line.names("tools.disabled"), ["terminal", "file"])

    def test_a_list_that_is_not_a_list_is_refused(self):
        settings = self._write("tools:\n  disabled:\n    terminal: true\n")
        with self.assertRaises(ConfigError):
            settings.names("tools.disabled")


class ChannelTests(unittest.TestCase):
    """One file describes an agent that answers in more than one place."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "config.yaml"
        self.path.write_text(
            "agent:\n"
            "  model: common-model\n"
            "tools:\n"
            "  disabled: [terminal]\n"
            "channels:\n"
            "  whatsapp:\n"
            "    agent:\n"
            "      model: whatsapp-model\n"
            "    tools:\n"
            "      disabled: [terminal, code_execution]\n",
            encoding="utf-8",
        )
        self.settings = Settings.load(self.path)

    def test_a_channel_setting_wins_for_that_channel(self):
        self.assertEqual(self.settings.for_channel("whatsapp").text("agent.model"), "whatsapp-model")

    def test_a_channel_setting_is_invisible_to_the_others(self):
        self.assertEqual(self.settings.for_channel("console").text("agent.model"), "common-model")

    def test_the_common_setting_is_untouched(self):
        self.assertEqual(self.settings.text("agent.model"), "common-model")

    def test_a_channel_inherits_what_it_does_not_override(self):
        whatsapp = self.settings.for_channel("whatsapp")
        self.assertEqual(whatsapp.names("tools.disabled"), ["terminal", "code_execution"])
        self.assertEqual(self.settings.names("tools.disabled"), ["terminal"])

    def test_a_block_is_merged_key_by_key(self):
        """Setting one key of a block must not silently drop the rest of it."""
        self.path.write_text(
            "stt:\n"
            "  provider: openai\n"
            "  language: ''\n"
            "channels:\n"
            "  whatsapp:\n"
            "    stt:\n"
            "      language: fr\n",
            encoding="utf-8",
        )
        block = Settings.load(self.path).for_channel("whatsapp").section("stt")
        self.assertEqual(block, {"provider": "openai", "language": "fr"})


class ConfigFileTests(unittest.TestCase):
    """The file is what `Config.load` actually reads."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        patch = mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.home)})
        patch.start()
        self.addCleanup(patch.stop)

    def _write(self, text: str) -> None:
        (self.home / "config.yaml").write_text(text, encoding="utf-8")

    def test_the_file_is_read_from_the_state_directory(self):
        self.assertEqual(config_path(self.home), self.home / "config.yaml")

    def test_an_unconfigured_agent_still_runs(self):
        config = Config.load()
        self.assertTrue(config.model)
        self.assertGreater(config.max_tool_iterations, 0)

    def test_the_file_sets_the_model(self):
        self._write("agent:\n  model: gpt-from-file\n")
        self.assertEqual(Config.load().model, "gpt-from-file")

    def test_the_file_beats_the_environment(self):
        self._write("agent:\n  model: gpt-from-file\n")
        with mock.patch.dict(os.environ, {"PILOTAGE_MODEL": "gpt-from-env"}):
            self.assertEqual(Config.load().model, "gpt-from-file")

    def test_the_environment_still_works_where_the_file_is_silent(self):
        self._write("agent:\n  history_turns: 5\n")
        with mock.patch.dict(os.environ, {"PILOTAGE_MODEL": "gpt-from-env"}):
            config = Config.load()
        self.assertEqual(config.model, "gpt-from-env")
        self.assertEqual(config.history_turns, 5)

    def test_the_operators_instructions_keep_the_formatting_note(self):
        self._write("agent:\n  instructions: Answer in French.\n")
        instructions = Config.load().instructions
        self.assertIn("Answer in French.", instructions)
        self.assertIn("WhatsApp formatting", instructions)

    def test_a_channel_can_be_loaded_on_its_own(self):
        self._write(
            "agent:\n"
            "  model: common-model\n"
            "channels:\n"
            "  whatsapp:\n"
            "    agent:\n"
            "      model: whatsapp-model\n"
        )
        self.assertEqual(Config.load().model, "common-model")
        self.assertEqual(Config.load(channel="whatsapp").model, "whatsapp-model")
        self.assertEqual(Config.load().for_channel("whatsapp").model, "whatsapp-model")

    def test_a_broken_file_stops_the_agent_rather_than_defaulting(self):
        self._write("agent:\n  model: [unclosed\n")
        with self.assertRaises(ConfigError):
            Config.load()

    def test_a_broken_file_exits_instead_of_starting(self):
        from pilotage import main

        self._write("agent:\n  model: [unclosed\n")
        with self.assertLogs("pilotage", level="ERROR"):
            self.assertEqual(main.main(["run"]), 1)

    def test_a_broken_channel_override_exits_instead_of_traceback(self):
        from pilotage import main

        self._write(
            "agent:\n"
            "  model: common-model\n"
            "channels:\n"
            "  whatsapp:\n"
            "    tools:\n"
            "      max_iterations: 0\n"
        )
        with self.assertLogs("pilotage", level="ERROR"):
            self.assertEqual(main.main(["run"]), 1)

    def test_non_positive_tool_output_caps_are_refused(self):
        for key in ("max_result_chars", "max_step_chars"):
            with self.subTest(key=key):
                self._write(f"tools:\n  {key}: 0\n")
                with self.assertRaises(ConfigError):
                    Config.load()


class RuntimeChannelTests(unittest.IsolatedAsyncioTestCase):
    async def test_whatsapp_runs_with_its_channel_configuration(self):
        from pilotage import main

        channel_config = Config.load(channel="whatsapp")

        seen = {}

        class FakeAgent:
            def __init__(self, config):
                seen["agent"] = config

        class FakeChannel:
            def __init__(self, config, handler, reset):
                seen["channel"] = config
                self.stopped = asyncio.Event()
                self.failure = None

            async def start(self):
                self.stopped.set()

            async def stop(self):
                pass

        with (
            mock.patch.object(main, "Agent", FakeAgent),
            mock.patch.object(main, "WhatsAppChannel", FakeChannel),
            mock.patch.object(main.auth, "read_credentials"),
        ):
            self.assertEqual(await main.command_run(channel_config), 0)

        self.assertIs(seen["agent"], channel_config)
        self.assertIs(seen["channel"], channel_config)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
