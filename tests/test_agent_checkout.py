from __future__ import annotations

import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pilotage.tools.file_operations import ShellFileOperations


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/install-agent-checkout.py"
HELPERS = runpy.run_path(str(SCRIPT))
asset_paths = HELPERS["asset_paths"]
install_assets = HELPERS["install_assets"]
share_skills = HELPERS["share_skills"]


class CheckoutAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.home = self.root / "home"
        self.source.mkdir()
        self.home.mkdir()
        self.write(self.source, ".gitignore", ".pilotage-agent/.env\n")
        self.write(self.source, ".pilotage-agent/config.yaml", "model: demo\n")
        self.write(self.source, ".pilotage-agent/SOUL.md", "Demo identity\n")
        self.write(self.source, ".pilotage-agent/skills/example/SKILL.md", "Demo skill\n")
        self.write(self.home, ".pilotage-agent/config.yaml", "bootstrap default\n")
        self.write(self.home, ".pilotage-agent/SOUL.md", "")
        self.write(self.home, ".pilotage-agent/.env", "KEEP=secret\n")

    @staticmethod
    def write(root, relative, content):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_fresh_overlay_includes_assets_but_not_git_metadata_or_runtime_data(self):
        self.write(self.source, ".git/config", "metadata")
        paths = asset_paths(self.source, self.home)
        self.assertIn(Path(".pilotage-agent/config.yaml"), paths)
        self.assertIn(Path(".pilotage-agent/skills/example/SKILL.md"), paths)
        self.assertNotIn(Path(".git/config"), paths)
        self.assertNotIn(Path(".pilotage-agent/.env"), paths)
        self.assertEqual((self.home / ".pilotage-agent/.env").read_text(), "KEEP=secret\n")

    def test_install_preserves_local_state_and_directory_modes(self):
        state = self.home / ".pilotage-agent"
        before = state.stat().st_mode
        paths = asset_paths(self.source, self.home)
        with patch("os.chown", create=True):
            install_assets(self.source, self.home, paths, 1000, 1001)
        self.assertEqual((state / ".env").read_text(), "KEEP=secret\n")
        self.assertEqual(state.stat().st_mode, before)
        self.assertEqual((state / "config.yaml").read_text(), "model: demo\n")
        self.assertEqual((state / "skills/example/SKILL.md").read_text(), "Demo skill\n")

    @unittest.skipUnless(shutil.which("git"), "Git is unavailable")
    def test_relocated_checkout_tracks_edits_to_live_assets(self):
        env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}

        def git(root, *args):
            return subprocess.run(
                ["git", "-c", "core.excludesFile=", "-c", "user.name=Test",
                 "-c", "user.email=test@example.invalid", "-C", str(root), *args],
                env=env, check=True, text=True, capture_output=True,
            ).stdout

        git(self.source, "init")
        git(self.source, "add", ".")
        git(self.source, "commit", "-m", "Initial assets")
        paths = asset_paths(self.source, self.home)
        with patch("os.chown", create=True):
            install_assets(self.source, self.home, paths, 1000, 1001)
        shutil.move(str(self.source / ".git"), self.home / ".git")
        self.assertEqual(git(self.home, "status", "--porcelain"), "")
        skill = self.home / ".pilotage-agent/skills/example/SKILL.md"
        skill.write_text("Edit from the running agent\n")
        self.assertIn(".pilotage-agent/skills/example/SKILL.md", git(self.home, "status", "--porcelain"))
        self.assertIn("Edit from the running agent", git(self.home, "diff", "--no-ext-diff", "--no-textconv"))
        self.assertNotIn(".env", git(self.home, "status", "--porcelain"))

    def test_runtime_and_account_files_cannot_be_supplied_by_repository(self):
        for name in (".pilotage-agent/.env", ".pilotage-agent/.runtime.lock",
                     ".pilotage-agent/codex-auth.json", ".pilotage-agent/profiles/other/config.yaml",
                     ".ssh/config", ".bashrc", "workspace/data.txt", ".gitconfig"):
            with self.subTest(name=name):
                path = self.write(self.source, name, "unsafe")
                with self.assertRaisesRegex(ValueError, "non-asset"):
                    asset_paths(self.source, self.home)
                path.unlink()
                parent = path.parent
                while parent != self.source and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent

    def test_existing_skill_edits_are_not_overwritten(self):
        target = self.write(self.home, ".pilotage-agent/skills/example/SKILL.md", "learned edit")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            asset_paths(self.source, self.home)
        self.assertEqual(target.read_text(), "learned edit")
        self.assertEqual((self.home / ".pilotage-agent/config.yaml").read_text(), "bootstrap default\n")

    def test_existing_checkout_is_not_reset(self):
        (self.home / ".git").mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            asset_paths(self.source, self.home)

    def test_required_config_and_identity_must_be_files(self):
        for name in (".pilotage-agent/config.yaml", ".pilotage-agent/SOUL.md", ".gitignore"):
            with self.subTest(name=name):
                target = self.source / name
                saved = target.read_text()
                target.unlink()
                with self.assertRaisesRegex(ValueError, "missing"):
                    asset_paths(self.source, self.home)
                target.write_text(saved)

    def test_symlinked_destination_is_rejected_without_touching_target(self):
        outside = self.root / "outside"
        outside.mkdir()
        link = self.home / ".pilotage-agent/skills"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Symlink creation is unavailable")
        with self.assertRaisesRegex(ValueError, "links"):
            asset_paths(self.source, self.home)
        self.assertEqual(list(outside.iterdir()), [])

    def test_symlinked_source_is_rejected(self):
        path = self.source / ".pilotage-agent/skills/link"
        try:
            path.symlink_to(self.source / ".pilotage-agent/config.yaml")
        except OSError:
            self.skipTest("Symlink creation is unavailable")
        with self.assertRaisesRegex(ValueError, "regular file"):
            asset_paths(self.source, self.home)

    def test_linked_bootstrap_setting_is_rejected(self):
        path = self.home / ".pilotage-agent/config.yaml"
        try:
            os.link(path, self.root / "linked-config")
        except OSError:
            self.skipTest("Hard links are unavailable")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            asset_paths(self.source, self.home)

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "geteuid")
                         and os.geteuid() == 0 and shutil.which("setfacl")
                         and shutil.which("git"), "Requires Linux root, Git and ACL support")
    def test_live_git_detects_agent_edits_and_both_users_can_replace_skill_files(self):
        operator_uid, agent_uid = 61000, 61001
        self.root.chmod(0o755)
        os.chown(self.home, operator_uid, agent_uid)
        self.home.chmod(0o755)
        state = self.home / ".pilotage-agent"
        os.chown(state, 0, agent_uid)
        state.chmod(0o3770)
        paths = asset_paths(self.source, self.home)
        install_assets(self.source, self.home, paths, operator_uid, agent_uid)
        share_skills(state / "skills", operator_uid, agent_uid)
        self.assertEqual((state / ".env").read_text(), "KEEP=secret\n")
        operator_home = self.root / "operator"
        operator_home.mkdir()
        os.chown(operator_home, operator_uid, operator_uid)
        operator_home.chmod(0o700)
        env = {**os.environ, "HOME": str(operator_home), "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_CONFIG_GLOBAL": os.devnull, "GIT_AUTHOR_NAME": "Test",
               "GIT_AUTHOR_EMAIL": "test@example.invalid", "GIT_COMMITTER_NAME": "Test",
               "GIT_COMMITTER_EMAIL": "test@example.invalid"}

        def run(uid, *args, input=None):
            return subprocess.run(args, cwd=self.home, env=env, user=uid,
                                  group=uid, extra_groups=[agent_uid], check=True,
                                  text=True, capture_output=True, input=input).stdout

        def git(*args):
            return run(operator_uid, "git", *args)

        git("init")
        (self.home / ".git").chmod(0o700)
        git("add", ".")
        git("commit", "-m", "Initial assets")
        old_head = git("rev-parse", "HEAD").strip()
        # Simulate Git replacing config with a permissive operator umask.
        run(operator_uid, "/usr/bin/python3", "-c", """
import os
from pathlib import Path
os.umask(0o002)
p = Path('.pilotage-agent/config.yaml')
p.unlink()
p.write_text('model: demo\\n')
assert p.stat().st_mode & 0o777 == 0o640
""")
        file = ".pilotage-agent/skills/example/SKILL.md"
        run(agent_uid, "/usr/bin/python3", "-c",
            "from pathlib import Path; Path('" + file + "').write_text('agent edit')")
        self.assertIn(file, git("status", "--porcelain"))
        self.assertIn("agent edit", git("diff", "--no-ext-diff", "--no-textconv"))
        git("add", file)
        git("commit", "-m", "Keep agent edit")
        # Exercise the checkout file replacement used by pull, without a network.
        git("restore", "--source=" + old_head, "--", file)
        run(agent_uid, "/usr/bin/python3", "-c",
            "from pathlib import Path; Path('" + file + "').write_text('agent edit after checkout')")
        def write_with_file_tool(uid, target, content):
            operations = ShellFileOperations(None, cwd=str(self.home))
            with patch.object(operations, "_exec") as execute:
                operations._atomic_write(str(target), content)
            command = execute.call_args.args[0]
            run(uid, "bash", "-c", "umask 077; " + command, input=content)

        # mktemp + chmod in the actual file tool previously masked the ACL and
        # left these files unreadable to the other account.
        for creator, reviewer in ((agent_uid, operator_uid), (operator_uid, agent_uid)):
            folder = f".pilotage-agent/skills/new-{creator}"
            write_with_file_tool(creator, self.home / folder / "SKILL.md", "new")
            git("add", folder)  # Must be able to read and stage agent-created files.
            run(reviewer, "/usr/bin/python3", "-c",
                f"from pathlib import Path; p=Path('{folder}/SKILL.md'); "
                "assert p.read_text()=='new'; p.unlink(); p.write_text('replaced')")
        private = self.root / "private-agent-data"
        private.mkdir()
        os.chown(private, agent_uid, agent_uid)
        private.chmod(0o700)
        write_with_file_tool(agent_uid, private / "secret", "private")
        self.assertEqual((private / "secret").stat().st_mode & 0o777, 0o600)
        run(agent_uid, "/usr/bin/python3", "-c", """
import os
from pathlib import Path
for p in (Path('.git'), Path('.pilotage-agent/config.yaml'), Path('.pilotage-agent/SOUL.md')):
    assert not os.access(p, os.W_OK)
assert not os.access('.git', os.R_OK)
for p in (Path('.git'), Path('.pilotage-agent/config.yaml')):
    try:
        p.rename(p.with_name(p.name + '-moved'))
    except PermissionError:
        pass
    else:
        raise AssertionError('agent renamed protected path')
""")


if __name__ == "__main__":
    unittest.main()
