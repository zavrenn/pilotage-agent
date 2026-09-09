"""The operator boundary and Linux ownership rules used by protected installs."""

import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from contextlib import redirect_stderr
from unittest import mock

from pilotage import deployment, env, main, profiles, runtime_lock, settings, update


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.managed = deployment.Deployment("operator", 1001)

    def test_manifest_must_be_root_owned_and_accounts_must_have_distinct_uids(self):
        manifest = mock.Mock()
        manifest.read_text.return_value = '{"operator":"operator"}'
        accounts = {"operator": SimpleNamespace(pw_name="operator", pw_uid=1001),
                    "agent": SimpleNamespace(pw_name="agent", pw_uid=1002)}
        with (
            mock.patch.object(deployment.sys, "platform", "linux"),
            mock.patch.object(deployment, "CHECKOUT", Path(deployment.__file__).resolve().parent.parent),
            mock.patch.object(deployment, "MANIFEST", manifest),
            mock.patch.dict(sys.modules, {"pwd": SimpleNamespace(getpwnam=accounts.__getitem__)}),
        ):
            for mode, uid in ((stat.S_IFREG | 0o666, 0), (stat.S_IFREG | 0o644, 1002), (stat.S_IFLNK | 0o777, 0)):
                manifest.lstat.return_value = SimpleNamespace(st_mode=mode, st_uid=uid)
                with self.assertRaises(PermissionError):
                    deployment.load()
            manifest.lstat.return_value = SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_uid=0)
            self.assertEqual(deployment.load(), self.managed)
            accounts["operator"].pw_uid = 1002
            with self.assertRaises(ValueError):
                deployment.load()

    def test_operator_controls_system_service_and_runtime_has_no_elevation(self):
        with mock.patch.object(deployment, "load", return_value=self.managed):
            with mock.patch.object(os, "geteuid", return_value=1001, create=True):
                self.assertEqual(deployment.system_command("systemctl"), ["systemctl"])
                self.assertEqual(deployment.system_command("systemctl", privileged=True), ["sudo", "--", "systemctl"])
            for uid in (0, 1002):
                with mock.patch.object(os, "geteuid", return_value=uid, create=True):
                    with self.assertRaises(PermissionError):
                        deployment.system_command("systemctl", privileged=True)

    def test_agent_state_commands_drop_operator_identity(self):
        with (
            mock.patch.object(deployment, "load", return_value=self.managed),
            mock.patch.object(os, "geteuid", return_value=1001, create=True),
            mock.patch.object(deployment.subprocess, "call", return_value=0) as run,
        ):
            self.assertEqual(deployment.as_agent(["--profile", "work", "doctor"]), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[:6], ["sudo", "-H", "-u", "agent", "--", "/usr/bin/env"])
            self.assertIn("-I", command)
            self.assertEqual(command[-3:], ["--profile", "work", "doctor"])

    def test_runtime_cannot_invoke_operator_commands(self):
        for command in ("restart", "update", "logs", "telegram", "whatsapp"):
            with (
                self.subTest(command=command),
                mock.patch.object(deployment, "load", return_value=self.managed),
                mock.patch.object(os, "geteuid", return_value=1002, create=True),
                mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(deployment.STATE)}, clear=False),
                mock.patch.object(main, "load_env_files", side_effect=AssertionError("loaded secrets")),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main.main([command]), 1)

    def test_state_commands_dispatch_before_loading_mutable_state(self):
        with (
            mock.patch.object(deployment, "load", return_value=self.managed),
            mock.patch.object(os, "geteuid", return_value=1001, create=True),
            mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(deployment.STATE)}, clear=False),
            mock.patch.object(deployment, "as_agent", return_value=7) as run,
            mock.patch.object(main, "load_env_files", side_effect=AssertionError("loaded secrets")),
        ):
            self.assertEqual(main.main(["--profile", "work", "status"]), 7)
            run.assert_called_once_with(["--profile", "work", "status"])

    def test_isolated_state_is_never_silently_replaced_by_production_state(self):
        with (
            mock.patch.object(deployment, "load", return_value=self.managed),
            mock.patch.dict(os.environ, {"PILOTAGE_HOME": "/isolated-test-state"}),
            mock.patch.object(deployment, "as_agent", side_effect=AssertionError("dispatched live work")),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main.main(["status"]), 1)
            self.assertEqual(profiles.default_state_root(), Path("/isolated-test-state"))

    def test_managed_selection_ignores_agent_written_sticky_profile(self):
        with (
            mock.patch.object(deployment, "load", return_value=self.managed),
            mock.patch.object(profiles, "_active_profile_path", side_effect=AssertionError("read mutable selection")),
        ):
            self.assertEqual(profiles.get_active_profile(), "default")

    def test_protected_config_and_env_overrides_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            state = Path(temporary)
            (state / ".env").write_text("UNTRUSTED_VALUE=should-not-load\n")
            with (
                mock.patch.object(deployment, "load", return_value=self.managed),
                mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(state)}, clear=True),
            ):
                for key in ("PILOTAGE_CONFIG", "PILOTAGE_ENV_FILE"):
                    with self.subTest(key=key), mock.patch.dict(os.environ, {key: str(state / "workspace/override")}):
                        with self.assertRaises(ValueError):
                            env.load_env_files()
                        with self.assertRaises(ValueError):
                            settings.config_path(state)
                        self.assertNotIn("UNTRUSTED_VALUE", os.environ)
                for key in ("PILOTAGE_CONFIG", "PILOTAGE_ENV_FILE"):
                    (state / ".env").write_text(f"UNTRUSTED_VALUE=should-not-load\n{key}={state / 'workspace/override'}\n")
                    with self.assertRaises(ValueError):
                        env.load_env_files()
                    self.assertNotIn("UNTRUSTED_VALUE", os.environ)

    def test_protected_policy_uses_selected_profile_and_never_repo_env_fallback(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            state = Path(temporary) / "profiles/work"
            state.mkdir(parents=True)
            with (
                mock.patch.object(deployment, "load", return_value=self.managed),
                mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(state),
                    "PILOTAGE_CONFIG": str(state / "config.yaml"),
                    "PILOTAGE_ENV_FILE": str(state / ".env")}, clear=True),
            ):
                self.assertEqual(env.candidate_env_files(), [state / ".env"])
                self.assertEqual(settings.config_path(state), state / "config.yaml")
                self.assertEqual(env.load_env_files(), [])

    def test_installer_parsing_never_applies_profile_environment(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / ".env"
            path.write_text('PATH="untrusted"\nPATH=duplicate\nPILOTAGE_CONFIG=elsewhere\n')
            before = dict(os.environ)
            values = env.read_env_values(path)
            self.assertEqual(values["PATH"], "untrusted")
            with self.assertRaises(ValueError):
                deployment.validate_policy_paths(path.parent, values)
            self.assertEqual(dict(os.environ), before)

    def test_profile_creation_is_available_only_to_operator(self):
        for uid, expected in ((1001, 0), (1002, 1)):
            with (
                self.subTest(uid=uid),
                mock.patch.object(deployment, "load", return_value=self.managed),
                mock.patch.object(os, "geteuid", return_value=uid, create=True),
                mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(deployment.STATE)}),
                mock.patch.object(profiles, "create_profile", return_value=Path("/work")) as create,
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main.main(["profile", "create", "work"]), expected)
                self.assertEqual(create.call_count, int(uid == 1001))

    def test_only_profile_locks_use_shared_protected_inode(self):
        with mock.patch.object(deployment, "load", return_value=self.managed):
            self.assertTrue(deployment.protected_lock(deployment.STATE / ".runtime.lock"))
            self.assertTrue(deployment.protected_lock(deployment.STATE / "profiles/work/.runtime.lock"))
            self.assertFalse(deployment.protected_lock(Path("/opt/pilotage-agent/.git/pilotage-update/.runtime.lock")))
            self.assertFalse(deployment.protected_lock(deployment.STATE / "workspace/.runtime.lock"))

    def test_managed_install_does_not_rewrite_agent_state_or_launcher(self):
        process = mock.Mock(pid=1234)
        process.wait.return_value = 0
        with (
            mock.patch.object(deployment, "load", return_value=self.managed),
            mock.patch.object(update.subprocess, "Popen", return_value=process) as run,
        ):
            self.assertEqual(update._install(Path("/opt/pilotage-agent"), lock_fds=(10, 11)), 0)
            self.assertEqual(run.call_args.args[0][-1], "--dependencies-only")
            self.assertEqual(run.call_args.kwargs["pass_fds"], (10, 11))

    def test_whatsapp_policy_written_by_operator_and_session_work_delegated(self):
        config = mock.Mock(state_dir=Path("/state"))
        with (
            mock.patch.object(main, "ProfileRuntimeLock"),
            mock.patch.object(main, "_prompt_whatsapp_configuration", return_value=([], {"PILOTAGE_ALLOWED_SENDERS": "123"})),
            mock.patch.object(main, "update_env_values") as save_env,
            mock.patch.object(main, "_save_channel_enabled", return_value=True) as save_enabled,
            mock.patch.object(deployment, "as_agent", return_value=0) as run,
        ):
            self.assertEqual(main._managed_whatsapp_setup(config, "work", Path("/env"), Path("/config"), frozenset()), 0)
            save_env.assert_called_once_with(Path("/env"), {"PILOTAGE_ALLOWED_SENDERS": "123"})
            run.assert_called_once_with(["--profile", "work", "whatsapp", "--pair-only"])
            self.assertEqual([call.args[-1] for call in save_enabled.call_args_list], [False, True])
            run.return_value = 1
            save_enabled.reset_mock()
            self.assertEqual(main._managed_whatsapp_setup(config, "work", Path("/env"), Path("/config"), frozenset()), 1)
            save_enabled.assert_called_once_with(Path("/config"), "whatsapp", False)


@unittest.skipUnless(sys.platform == "linux", "requires POSIX file modes")
class LinuxDeploymentTests(unittest.TestCase):
    def test_restrictive_update_permissions_are_repaired_without_exposing_git_or_link_targets(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary) / "checkout"
            root.mkdir()
            previous = os.umask(0o077)
            try:
                (root / "package").mkdir()
                source = root / "package/module.py"
                source.write_text("value = 1\n")
                executable = root / "tool"
                executable.write_text("#!/bin/sh\nexit 0\n")
                executable.chmod(0o700)
                (root / ".git").mkdir()
                secret = root / ".git/config"
                secret.write_text("private")
                linked = Path(temporary) / "outside"
                linked.write_text("outside")
                (root / "symlink").symlink_to(linked)
            finally:
                os.umask(previous)
            deployment.make_runtime_readable(root)
            self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o644)
            self.assertEqual(stat.S_IMODE((root / "package").stat().st_mode), 0o755)
            self.assertEqual(stat.S_IMODE(executable.stat().st_mode), 0o755)
            self.assertEqual(stat.S_IMODE((root / ".git").stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(secret.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(linked.stat().st_mode), 0o600)

    def test_operator_can_create_protected_profile_with_readable_policy_and_shared_lock(self):
        import grp

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            state = Path(temporary)
            parent = state / "profiles"
            parent.mkdir()
            parent.chmod(0o2750)
            checkout = state / "checkout"
            managed = deployment.Deployment("operator", os.geteuid())
            with (
                mock.patch.object(deployment, "load", return_value=managed),
                mock.patch.object(deployment, "STATE", state),
                mock.patch.object(deployment, "CHECKOUT", checkout),
                mock.patch.object(grp, "getgrnam", return_value=SimpleNamespace(gr_gid=os.getgid())),
                mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(state)}, clear=True),
                mock.patch.object(profiles.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run,
            ):
                work = profiles.create_profile("work")
                self.assertEqual(stat.S_IMODE(work.stat().st_mode), 0o3770)
                self.assertEqual(work.stat().st_uid, managed.operator_uid)
                for name in (".env", "config.yaml", "SOUL.md"):
                    self.assertEqual(stat.S_IMODE((work / name).stat().st_mode), 0o640)
                    self.assertEqual((work / name).stat().st_gid, os.getgid())
                self.assertFalse((work / "workspace").exists())
                lock = runtime_lock.ProfileRuntimeLock(work)
                lock.acquire()
                lock.release()
                self.assertEqual(stat.S_IMODE((work / ".runtime.lock").stat().st_mode), 0o660)
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0], ["sudo", "--", "systemctl", "enable", "pilotage-agent@work.service"])
                self.assertFalse(settings.Settings.load(work / "config.yaml").flag("whatsapp.enabled", True))
                self.assertEqual(profiles._configured_bridge_port(work), 8766)
                second = profiles.create_profile("personal")
                self.assertEqual(profiles._configured_bridge_port(second), 8767)
                # Runtime updates and provisioning use the same operator lock.
                updating = runtime_lock.ProfileRuntimeLock(checkout / ".git/pilotage-update")
                updating.acquire()
                try:
                    with self.assertRaises(runtime_lock.RuntimeAlreadyRunning):
                        profiles.create_profile("blocked")
                finally:
                    updating.release()
                self.assertFalse((parent / "blocked").exists())

    def test_operator_settings_updates_keep_runtime_read_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text("whatsapp:\n  enabled: false\n")
            path.chmod(0o640)
            settings.set_channel_enabled(path, "whatsapp", True)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
            path = Path(temporary) / ".env"
            path.write_text("A=old\n")
            path.chmod(0o640)
            env.update_env_values(path, {"A": "new"})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)

    def test_protected_lock_preserves_mode_and_rejects_replacement_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".runtime.lock"
            path.touch()
            path.chmod(0o660)
            managed = deployment.Deployment("operator", os.geteuid())
            with (
                mock.patch.object(runtime_lock, "protected_lock", return_value=True),
                mock.patch.object(deployment, "load", return_value=managed),
            ):
                lock = runtime_lock.ProfileRuntimeLock(path.parent)
                lock.acquire()
                self.assertTrue(runtime_lock.runtime_lock_is_held(path.parent))
                lock.release()
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o660)
                alias = path.with_name("alias")
                path.rename(alias)
                path.symlink_to(alias)
                with self.assertRaises(runtime_lock.RuntimeLockError):
                    lock.acquire()
                path.unlink()
                os.link(alias, path)
                with self.assertRaises(runtime_lock.RuntimeLockError):
                    lock.acquire()

    @unittest.skipUnless(hasattr(os, "geteuid") and os.geteuid() == 0, "requires root in disposable Linux container")
    def test_actual_second_uid_cannot_replace_policy_but_can_write_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o3777)
            policy = root / "config.yaml"
            policy.write_text("protected")
            os.chown(policy, 65533, 65534)
            os.chmod(policy, 0o644)
            result = subprocess.run([sys.executable, "-I", "-B", "-c", '''
import pathlib, sys
p = pathlib.Path(sys.argv[1])
for action in (lambda: p.write_text("bad"), lambda: p.unlink(), lambda: p.rename(p.with_suffix(".old")),
               lambda: p.parent.chmod(0o777)):
    try: action()
    except PermissionError: continue
    raise AssertionError("protected policy changed")
own = p.parent / "skill.md"
own.write_text("learned")
own.write_text("updated")
own.unlink()
''', str(policy)], user=65534, group=65534, extra_groups=[], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(policy.read_text(), "protected")
