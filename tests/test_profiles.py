from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pilotage import profiles
from pilotage.codex import auth
from pilotage.config import Config
from pilotage.runtime_lock import ProfileRuntimeLock
from pilotage.settings import Settings


def _credentials(access: str = "access", refresh: str = "refresh") -> auth.Credentials:
    return auth.Credentials(
        access_token=access,
        refresh_token=refresh,
        base_url="https://example.invalid/codex",
        last_refresh="now",
    )


class ProfileTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        patch = mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.root)})
        patch.start()
        self.addCleanup(patch.stop)

    def test_profile_creation_starts_isolated(self):
        path = profiles.create_profile("Work")

        self.assertEqual(path, self.root / "profiles" / "work")
        for directory in (
            "memories",
            "sessions",
            "skills",
            "logs",
            "workspace",
            "cron",
            "whatsapp",
            "media",
            "home",
        ):
            self.assertTrue((path / directory).is_dir(), directory)
        self.assertTrue((path / ".env").is_file())
        self.assertTrue((path / "config.yaml").is_file())
        settings = Settings.load(path / "config.yaml").for_channel("whatsapp")
        self.assertEqual(settings.count("whatsapp.bridge_port", 0), 8766)
        self.assertFalse(settings.flag("whatsapp.enabled", True))
        self.assertFalse(settings.flag("telegram.enabled", True))
        self.assertEqual(settings.text("display.language"), "fr")
        self.assertEqual(settings.get("timezone"), "")
        self.assertFalse((path / "codex-auth.json").exists())

    def test_each_profile_receives_an_explicit_unique_bridge_port(self):
        (self.root / "config.yaml").write_text(
            "whatsapp:\n  bridge_port: 8766\n",
            encoding="utf-8",
        )

        first = profiles.create_profile("first")
        second = profiles.create_profile("second")

        first_settings = Settings.load(first / "config.yaml").for_channel("whatsapp")
        second_settings = Settings.load(second / "config.yaml").for_channel("whatsapp")
        self.assertEqual(first_settings.count("whatsapp.bridge_port", 0), 8767)
        self.assertEqual(second_settings.count("whatsapp.bridge_port", 0), 8768)
        self.assertIsNone(first_settings.get("agent.model"))

    def test_broken_existing_profile_config_stops_port_allocation(self):
        existing = profiles.create_profile("existing")
        (existing / "config.yaml").write_text("whatsapp: [broken\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Cannot allocate a profile port"):
            profiles.create_profile("new")

        self.assertFalse((self.root / "profiles" / "new").exists())

    def test_partial_profile_creation_is_removed(self):
        original = Path.write_text

        def fail_config(path, *args, **kwargs):
            if path.name == "config.yaml":
                raise OSError("disk full")
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "write_text", fail_config):
            with self.assertRaisesRegex(OSError, "disk full"):
                profiles.create_profile("partial")

        self.assertFalse((self.root / "profiles" / "partial").exists())

    def test_linked_profile_state_is_never_listed_or_activated(self):
        outside = self.root / "outside"
        outside.mkdir()
        root = self.root / "profiles"
        root.mkdir()
        linked = root / "linked"
        try:
            linked.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"directory symlinks are unavailable: {exc}")

        self.assertFalse(profiles.profile_exists("linked"))
        self.assertNotIn("linked", [item.name for item in profiles.list_profiles()])
        with self.assertRaises(FileNotFoundError):
            profiles.activate_for_process("linked")
        with self.assertRaises(FileExistsError):
            profiles.create_profile("linked")
        with self.assertRaises(ValueError):
            profiles.delete_profile("linked")
        self.assertTrue(outside.is_dir())

    def test_profile_names_cannot_escape_the_root(self):
        for name in ("../outside", "a/b", ".hidden", "pilotage", "tmp"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                profiles.create_profile(name)

    def test_sticky_profile_selection_resolves_back_to_the_main_root(self):
        path = profiles.create_profile("enterprise")
        profiles.set_active_profile("enterprise")
        name, selected = profiles.activate_for_process()

        self.assertEqual((name, selected), ("enterprise", path))
        self.assertEqual(profiles.default_state_root(), self.root)
        self.assertEqual(os.environ["PILOTAGE_HOME"], str(path))

    def test_all_runtime_paths_follow_the_selected_profile(self):
        path = profiles.create_profile("personal")
        profiles.activate_for_process("personal")
        config = Config.load()

        self.assertEqual(config.state_dir, path)
        self.assertEqual(config.session_dir, path / "whatsapp")
        self.assertEqual(config.media_dir, path / "media")
        self.assertEqual(config.conversations_path, path / "conversations.db")
        self.assertEqual(config.credentials_path, path / "codex-auth.json")
        self.assertEqual(config.main_credentials_path, self.root / "codex-auth.json")
        self.assertEqual(config.bridge_port, 8766)

    def test_listing_marks_the_sticky_profile(self):
        profiles.create_profile("personal")
        profiles.create_profile("enterprise")
        profiles.set_active_profile("enterprise")

        listed = profiles.list_profiles()
        self.assertEqual([item.name for item in listed], ["default", "enterprise", "personal"])
        self.assertEqual([item.name for item in listed if item.is_active], ["enterprise"])

    def test_delete_refuses_a_running_profile(self):
        profile = profiles.create_profile("busy")
        lock = ProfileRuntimeLock(profile)
        lock.acquire()
        self.addCleanup(lock.release)

        with self.assertRaisesRegex(ValueError, "Stop the profile"):
            profiles.delete_profile("busy")
        self.assertTrue(profile.is_dir())

        lock.release()
        profiles.delete_profile("busy")
        self.assertFalse(profile.exists())
    def test_deleting_the_active_profile_returns_to_default(self):
        path = profiles.create_profile("personal")
        profiles.set_active_profile("personal")

        self.assertEqual(profiles.delete_profile("personal"), path)
        self.assertFalse(path.exists())
        self.assertEqual(profiles.get_active_profile(), "default")
        self.assertFalse((self.root / "active_profile").exists())

    def test_cli_profile_management_uses_the_same_contract(self):
        from pilotage import main

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main.main(["profile", "create", "team"]), 0)
            self.assertEqual(main.main(["profile", "use", "team"]), 0)
            self.assertEqual(main.main(["profile", "show"]), 0)
            self.assertEqual(main.main(["profile", "delete", "team", "--yes"]), 0)

        self.assertIn("team", output.getvalue())
        self.assertEqual(profiles.get_active_profile(), "default")

    def test_profile_is_selected_before_its_configuration_is_loaded(self):
        from pilotage import main

        path = profiles.create_profile("team")
        (path / "config.yaml").write_text("agent:\n  model: profile-model\n", encoding="utf-8")
        seen = {}

        def fake_status(config, profile_name):
            seen["config"] = config
            seen["profile_name"] = profile_name
            return 0

        with mock.patch.object(main, "command_status", fake_status):
            self.assertEqual(main.main(["-p", "team", "status"]), 0)

        self.assertEqual(seen["config"].state_dir, path)
        self.assertEqual(seen["config"].model, "profile-model")
        self.assertEqual(seen["profile_name"], "team")

    def test_interactive_terminal_conversation_is_not_a_command(self):
        from pilotage import main

        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main.main(["ask", "hello"])
        self.assertEqual(raised.exception.code, 2)


class ProfileAuthenticationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.profile = self.root / "profiles" / "work"
        self.profile.mkdir(parents=True)
        self.main_path = self.root / "codex-auth.json"
        self.profile_path = self.profile / "codex-auth.json"

    def test_missing_profile_auth_uses_main_auth(self):
        auth.write_credentials(self.main_path, _credentials(refresh="main"))

        found = auth.read_credentials(self.profile_path, fallback_path=self.main_path)
        self.assertEqual(found.refresh_token, "main")

    def test_profile_auth_shadows_main_auth(self):
        auth.write_credentials(self.main_path, _credentials(refresh="main"))
        auth.write_credentials(self.profile_path, _credentials(refresh="profile"))

        found = auth.read_credentials(self.profile_path, fallback_path=self.main_path)
        self.assertEqual(found.refresh_token, "profile")

    def test_broken_profile_auth_does_not_fall_back(self):
        auth.write_credentials(self.main_path, _credentials(refresh="main"))
        self.profile_path.write_text(json.dumps({"wrong": "shape"}), encoding="utf-8")

        with self.assertRaises(auth.AuthError):
            auth.read_credentials(self.profile_path, fallback_path=self.main_path)

    def test_refresh_of_borrowed_auth_updates_its_main_source(self):
        auth.write_credentials(self.main_path, _credentials(access="expired", refresh="old"))
        refreshed = _credentials(access="fresh", refresh="rotated")

        with (
            mock.patch.object(auth, "access_token_is_expiring", return_value=True),
            mock.patch.object(auth, "refresh_credentials", return_value=refreshed),
        ):
            found = auth.resolve_credentials(
                self.profile_path,
                fallback_path=self.main_path,
            )

        self.assertEqual(found.refresh_token, "rotated")
        self.assertEqual(auth.read_credentials(self.main_path).refresh_token, "rotated")
        self.assertFalse(self.profile_path.exists())

    def test_login_writes_the_profile_without_replacing_main_auth(self):
        from pilotage import main

        auth.write_credentials(self.main_path, _credentials(refresh="main"))
        with mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.profile)}):
            config = Config.load()
            with (
                mock.patch.object(
                    main.auth,
                    "device_code_login",
                    return_value=_credentials(refresh="profile"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main.command_login(config), 0)

        self.assertEqual(auth.read_credentials(self.profile_path).refresh_token, "profile")
        self.assertEqual(auth.read_credentials(self.main_path).refresh_token, "main")


if __name__ == "__main__":
    unittest.main()
